from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from outer.agent_client import AgentClientError, build_agent_client
from outer.aggregate_eval import (
    build_trial_metric_summary,
    build_warmup_summary,
    compute_benchmark_summary,
    compute_family_utility_online,
    compute_proxy_quality_summary,
)
from outer.capabilities import build_capability_registry
from outer.execution import execute_family_aware_trial, execute_warmup_paired_probe_bundle
from outer.family_memory import (
    build_family_leaderboard,
    build_family_memory_view,
    build_family_round_record,
    build_family_summary_table,
    write_family_memory_artifacts,
)
from outer.prompt_builder import (
    build_train_prompt_bundle,
    build_validator_prompt_bundle,
    build_validator_repair_prompt_bundle,
)
from outer.schema import ProposalError, SearchSpace, clone_with_locked_family, compute_candidate_hash, resolve_proposal
from outer.validator_role import (
    ValidatorRoleError,
    build_builtin_validator_role_output,
    build_validation_error_report,
    normalize_validator_role_output,
)
from training.validation_protocols import VALIDATOR_FAMILY_PRESET


@dataclass
class StopConfig:
    max_trials: int = 20
    max_failures: int = 6
    max_runtime_hours: float = 8.0
    patience: int = 5


class AutoValiSearchController:
    def __init__(
        self,
        workdir: str,
        search_space: SearchSpace,
        stop_config: StopConfig,
        *,
        agent_mode: str = 'heuristic',
        agent_model: str = 'gpt-5.4',
        agent_base_url: str = 'https://www.autodl.art/api/v1',
        agent_api_key_env: str = 'AUTODL_API_KEY',
        agent_timeout_sec: float = 180.0,
        agent_max_attempts: int = 4,
        agent_backend: Optional[str] = None,
        agent_server_url: Optional[str] = None,
        agent_model_name: Optional[str] = None,
        agent_max_retries: Optional[int] = None,
        agent_backend_metadata: Optional[Dict[str, Any]] = None,
        val_agent_mode: Optional[str] = None,
        val_agent_model: Optional[str] = None,
        val_agent_base_url: Optional[str] = None,
        val_agent_api_key_env: Optional[str] = None,
        val_agent_timeout_sec: Optional[float] = None,
        val_agent_max_attempts: Optional[int] = None,
        val_agent_backend: Optional[str] = None,
        val_agent_server_url: Optional[str] = None,
        val_agent_model_name: Optional[str] = None,
        val_agent_max_retries: Optional[int] = None,
        train_agent_mode: Optional[str] = None,
        train_agent_model: Optional[str] = None,
        train_agent_base_url: Optional[str] = None,
        train_agent_api_key_env: Optional[str] = None,
        train_agent_timeout_sec: Optional[float] = None,
        train_agent_max_attempts: Optional[int] = None,
        train_agent_backend: Optional[str] = None,
        train_agent_server_url: Optional[str] = None,
        train_agent_model_name: Optional[str] = None,
        train_agent_max_retries: Optional[int] = None,
        python_executable: str = 'python',
        design_only: bool = False,
        top_k: int = 5,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir = self.workdir / 'rounds'
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.workdir / 'controller.log'
        self.search_space = search_space
        self.stop_config = stop_config
        self.design_only = design_only
        self.top_k = top_k
        self.python_executable = python_executable
        self.started_at = time.time()
        self.fixed_config = dict(search_space.fixed_config)
        self.max_family_rounds = max(int(self.fixed_config.get('max_family_rounds', 1)), 1)
        self.family_memory_top_k = max(int(self.fixed_config.get('family_memory_top_k', 3)), 1)
        self.family_memory_failure_k = max(int(self.fixed_config.get('family_memory_failure_k', 3)), 1)
        self.trial_agent_client = build_agent_client(
            agent_mode=train_agent_mode or agent_mode,
            seed=int(self.fixed_config['seed']),
            agent_backend=train_agent_backend if train_agent_backend is not None else agent_backend,
            agent_server_url=train_agent_server_url if train_agent_server_url is not None else agent_server_url,
            agent_model_name=train_agent_model_name if train_agent_model_name is not None else agent_model_name,
            agent_model=train_agent_model or agent_model,
            agent_base_url=train_agent_base_url or agent_base_url,
            agent_api_key_env=train_agent_api_key_env or agent_api_key_env,
            agent_timeout_sec=float(train_agent_timeout_sec if train_agent_timeout_sec is not None else agent_timeout_sec),
            agent_max_attempts=int(train_agent_max_attempts if train_agent_max_attempts is not None else agent_max_attempts),
            agent_max_retries=train_agent_max_retries if train_agent_max_retries is not None else agent_max_retries,
            agent_backend_metadata=agent_backend_metadata,
        )
        self.validator_agent_client = build_agent_client(
            agent_mode=val_agent_mode or agent_mode,
            seed=int(self.fixed_config['seed']),
            agent_backend=val_agent_backend if val_agent_backend is not None else agent_backend,
            agent_server_url=val_agent_server_url if val_agent_server_url is not None else agent_server_url,
            agent_model_name=val_agent_model_name if val_agent_model_name is not None else agent_model_name,
            agent_model=val_agent_model or agent_model,
            agent_base_url=val_agent_base_url or agent_base_url,
            agent_api_key_env=val_agent_api_key_env or agent_api_key_env,
            agent_timeout_sec=float(val_agent_timeout_sec if val_agent_timeout_sec is not None else agent_timeout_sec),
            agent_max_attempts=int(val_agent_max_attempts if val_agent_max_attempts is not None else agent_max_attempts),
            agent_max_retries=val_agent_max_retries if val_agent_max_retries is not None else agent_max_retries,
            agent_backend_metadata=agent_backend_metadata,
        )
        self.family_records: List[Dict[str, Any]] = []
        self._write_json(
            self.workdir / 'run_config.json',
            {
                'project_name': 'AutoValiSearch',
                'stage': 'Stage 2',
                'fixed_config': self.fixed_config,
                'stop_config': asdict(self.stop_config),
                'design_only': self.design_only,
                'max_family_rounds': self.max_family_rounds,
                'family_memory_top_k': self.family_memory_top_k,
                'family_memory_failure_k': self.family_memory_failure_k,
                'backend': {
                    'validator_role': self.validator_agent_client.describe_backend(),
                    'trial_role': self.trial_agent_client.describe_backend(),
                },
            },
        )
        self._write_json(self.workdir / 'capability_registry.json', build_capability_registry())

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')

    def _write_jsonl(self, path: Path, records: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

    def log(self, message: str) -> None:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        line = f'[{timestamp}] {message}'
        print(line)
        with open(self.log_path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    def _warmup_budget_summary(self, warmup_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            'warmup_probe_count': len({json.dumps(record.get('probe_config', {}), sort_keys=True) for record in warmup_records}),
            'warmup_child_runs': len(warmup_records),
            'warmup_successful_child_runs': len([record for record in warmup_records if record.get('status') == 'ok']),
        }

    def _family_budget_summary(self, history: List[Dict[str, Any]], failures: List[Dict[str, Any]], invalid_count: int, execution_failure_count: int) -> Dict[str, Any]:
        return {
            'max_trials': self.stop_config.max_trials,
            'completed_trials': len([record for record in history if record.get('status') == 'ok']),
            'terminal_rounds': len(history) + len(failures),
            'proposal_invalid_count': invalid_count,
            'execution_failure_count': execution_failure_count,
        }

    def _build_warmup_probe_configs(self) -> List[Dict[str, Any]]:
        tunables = self.search_space.tunables
        lr_values = list(tunables['lr'])
        lambdap_values = list(tunables['lambdap'])
        epochp_values = list(tunables['epochp'])
        wd_values = list(tunables['weight_decay'])
        candidates = [
            {'lr': lr_values[0], 'lambdap': lambdap_values[min(1, len(lambdap_values) - 1)], 'epochp': epochp_values[0], 'weight_decay': wd_values[0]},
            {'lr': lr_values[0], 'lambdap': lambdap_values[min(2, len(lambdap_values) - 1)], 'epochp': epochp_values[-1], 'weight_decay': wd_values[min(1, len(wd_values) - 1)]},
            {'lr': lr_values[min(1, len(lr_values) - 1)], 'lambdap': lambdap_values[0], 'epochp': epochp_values[min(1, len(epochp_values) - 1)], 'weight_decay': wd_values[-1]},
            {'lr': lr_values[-1], 'lambdap': lambdap_values[-1], 'epochp': epochp_values[0], 'weight_decay': wd_values[0]},
        ]
        unique = []
        seen = set()
        for candidate in candidates:
            key = json.dumps(candidate, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique[: max(int(self.fixed_config.get('warmup_probe_count', 3)), 0)]

    def _effective_round_count(self) -> int:
        if self.fixed_config.get('validator_family_spec') is not None:
            return 1
        if str(self.fixed_config.get('validator_preset')) == VALIDATOR_FAMILY_PRESET:
            return 1
        if self.design_only:
            return 1
        return self.max_family_rounds

    def _prepare_round_dirs(self, round_index: int) -> Dict[str, Path]:
        round_dir = self.rounds_dir / f'round_{round_index:02d}'
        dirs = {
            'round_dir': round_dir,
            'trials_dir': round_dir / 'trials',
            'warmup_dir': round_dir / 'warmup',
            'validator_dir': round_dir / 'validator_design',
            'prompts_dir': round_dir / 'prompts',
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def _run_warmup(self, *, round_index: int, dirs: Mapping[str, Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        probe_configs = self._build_warmup_probe_configs()
        all_records: List[Dict[str, Any]] = []
        for index, probe_config in enumerate(probe_configs, start=1):
            probe_dir = dirs['warmup_dir'] / f'probe_{index:02d}'
            probe_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(probe_dir / 'probe_config.json', probe_config)
            records = execute_warmup_paired_probe_bundle(
                python_executable=self.python_executable,
                fixed_config=self.fixed_config,
                probe_config=probe_config,
                probe_dir=probe_dir,
            )
            all_records.extend(records)
            self._write_json(probe_dir / 'probe_results.json', {'records': records})
        summary = build_warmup_summary(all_records)
        summary['probe_configs'] = probe_configs
        summary['round_index'] = round_index
        self._write_json(dirs['round_dir'] / 'warmup_summary.json', summary)
        self._write_jsonl(dirs['round_dir'] / 'warmup_records.jsonl', all_records)
        return all_records, summary

    def _build_round_family_memory_view(self, round_index: int, dirs: Mapping[str, Path]) -> Dict[str, Any]:
        view = build_family_memory_view(
            self.family_records,
            top_k=self.family_memory_top_k,
            failure_k=self.family_memory_failure_k,
            current_round_index=round_index,
        )
        self._write_json(dirs['round_dir'] / 'family_memory_view.json', view)
        return view

    def _lock_family(self, *, round_index: int, dirs: Mapping[str, Path], warmup_summary: Mapping[str, Any], family_memory_view: Mapping[str, Any]) -> Dict[str, Any]:
        validator_dir = dirs['validator_dir']
        if self.fixed_config.get('validator_family_spec') is not None:
            payload = {
                'status': 'ok',
                'source': 'prelocked',
                'validator_family_spec': dict(self.fixed_config['validator_family_spec']),
            }
            self._write_json(validator_dir / 'validator_role_final_status.json', payload)
            return payload
        if str(self.fixed_config.get('validator_preset')) == VALIDATOR_FAMILY_PRESET:
            proposal = build_builtin_validator_role_output(
                validator_role_id=f'validator_role_round_{round_index:02d}',
                validator_protocol='validator_family',
                validator_preset=self.fixed_config.get('validator_preset'),
            )
            normalized = normalize_validator_role_output(
                proposal,
                validator_protocol='validator_family',
                validator_preset=self.fixed_config.get('validator_preset'),
            )
            payload = {
                'status': 'ok',
                'source': 'builtin',
                'validator_family_recipe': dict(normalized['validator_family_recipe']),
                'validator_family_spec': dict(normalized['validator_family_spec']),
            }
            self._write_json(validator_dir / 'validator_family_recipe.json', payload['validator_family_recipe'])
            self._write_json(validator_dir / 'validator_family_spec.json', payload['validator_family_spec'])
            self._write_json(validator_dir / 'validator_role_final_status.json', payload)
            return payload
        prompt_bundle = build_validator_prompt_bundle(
            validator_role_id=f'validator_role_round_{round_index:02d}',
            fixed_config=self.fixed_config,
            warmup_summary=warmup_summary,
            family_memory_view=family_memory_view,
            round_index=round_index,
        )
        self._write_json(validator_dir / 'validator_role_prompt_bundle.json', prompt_bundle)
        attempts = int(self.fixed_config.get('design_retry_limit', 3))
        previous_output: Mapping[str, Any] | str | None = None
        for attempt in range(1, attempts + 1):
            if attempt == 1:
                current_prompt = prompt_bundle
            else:
                current_prompt = build_validator_repair_prompt_bundle(
                    validator_role_id=f'validator_role_round_{round_index:02d}',
                    previous_output=previous_output or {},
                    validation_error_report=json.loads((validator_dir / f'validator_role_validation_error_attempt_{attempt - 1:02d}.json').read_text(encoding='utf-8')),
                    family_memory_view=family_memory_view,
                    round_index=round_index,
                )
                self._write_json(validator_dir / f'validator_role_prompt_bundle_attempt_{attempt:02d}.json', current_prompt)
            started = time.time()
            raw_output = self.validator_agent_client.propose_validator(
                f'validator_role_round_{round_index:02d}',
                current_prompt,
                validator_protocol='validator_family',
                validator_preset=self.fixed_config.get('validator_preset'),
            )
            latency = round(time.time() - started, 6)
            self._write_json(validator_dir / f'validator_role_raw_output_attempt_{attempt:02d}.json', raw_output)
            self._write_json(validator_dir / f'validator_role_runtime_attempt_{attempt:02d}.json', {'latency_sec': latency})
            previous_output = raw_output
            try:
                normalized = normalize_validator_role_output(
                    raw_output,
                    validator_protocol='validator_family',
                    validator_preset=self.fixed_config.get('validator_preset'),
                )
            except (ValidatorRoleError, AgentClientError) as exc:
                report = build_validation_error_report(exc)
                self._write_json(validator_dir / f'validator_role_validation_error_attempt_{attempt:02d}.json', report)
                continue
            payload = {
                'status': 'ok',
                'source': 'llm',
                'attempts': attempt,
                'validator_family_recipe': dict(normalized['validator_family_recipe']),
                'validator_family_spec': dict(normalized['validator_family_spec']),
            }
            self._write_json(validator_dir / 'validator_family_recipe.json', payload['validator_family_recipe'])
            self._write_json(validator_dir / 'validator_family_spec.json', payload['validator_family_spec'])
            self._write_json(validator_dir / 'validator_role_final_status.json', payload)
            return payload
        payload = {'status': 'validator_design_failed', 'attempts': attempts}
        self._write_json(validator_dir / 'validator_role_final_status.json', payload)
        raise ValidatorRoleError('validator_family design failed after bounded retries')

    def _top_best_trials(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        successful = [record for record in history if record.get('status') == 'ok']
        ordered = sorted(successful, key=lambda item: float(item.get('selection_score', float('-inf'))), reverse=True)
        return ordered[: self.top_k]

    def _recent_failures(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return failures[-self.top_k :]

    def _stop_reason(self, *, started_at: float, history: List[Dict[str, Any]], failures: List[Dict[str, Any]], execution_failure_count: int, no_improve_rounds: int) -> Optional[str]:
        runtime_hours = (time.time() - started_at) / 3600.0
        if len(history) + len(failures) >= self.stop_config.max_trials:
            return 'max_trials'
        if execution_failure_count >= self.stop_config.max_failures:
            return 'max_failures'
        if runtime_hours >= self.stop_config.max_runtime_hours:
            return 'max_runtime_hours'
        if no_improve_rounds >= self.stop_config.patience:
            return 'patience'
        return None

    def _record_failure(self, path: Path, failures: List[Dict[str, Any]], payload: Mapping[str, Any]) -> None:
        failures.append(dict(payload))
        self._write_jsonl(path, failures)


    def _run_family_search(self, *, round_index: int, dirs: Mapping[str, Path], locked_family_spec: Mapping[str, Any], family_utility_summary: Mapping[str, Any] | None) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        invalid_count = 0
        execution_failure_count = 0
        no_improve_rounds = 0
        best_trial: Optional[Dict[str, Any]] = None
        stop_reason: Optional[str] = None
        if self.design_only:
            return {
                'history': history,
                'failures': failures,
                'invalid_count': invalid_count,
                'execution_failure_count': execution_failure_count,
                'best_trial': best_trial,
                'stop_reason': 'design_only',
            }
        locked_search_space = clone_with_locked_family(self.search_space, locked_family_spec)
        forbidden_hashes = set()
        started_at = time.time()
        trial_index = 0
        while True:
            stop_reason = self._stop_reason(
                started_at=started_at,
                history=history,
                failures=failures,
                execution_failure_count=execution_failure_count,
                no_improve_rounds=no_improve_rounds,
            )
            if stop_reason is not None:
                self.log(f'Round {round_index}: stopping family-aware search due to {stop_reason}')
                break
            trial_index += 1
            trial_id = f'trial_{trial_index:04d}'
            prompt_bundle = build_train_prompt_bundle(
                proposal_id=trial_id,
                fixed_config=locked_search_space.fixed_config,
                tunables=locked_search_space.tunables,
                locked_family_spec=locked_family_spec,
                family_utility_summary=family_utility_summary,
                best_trials=self._top_best_trials(history),
                recent_failures=self._recent_failures(failures),
                round_index=round_index,
            )
            self._write_json(dirs['prompts_dir'] / f'{trial_id}_prompt_bundle.json', prompt_bundle)
            try:
                proposal = self.trial_agent_client.propose_trial(
                    trial_id,
                    prompt_bundle,
                    locked_search_space,
                    compute_candidate_hash,
                    set(filter(None, forbidden_hashes)),
                )
                resolved = resolve_proposal(proposal, locked_search_space, str(dirs['trials_dir'] / trial_id))
            except (ProposalError, AgentClientError, json.JSONDecodeError) as exc:
                invalid_count += 1
                no_improve_rounds += 1
                failure = {
                    'trial_id': trial_id,
                    'status': 'invalid',
                    'error_type': exc.__class__.__name__,
                    'error_message': str(exc),
                }
                self._record_failure(dirs['round_dir'] / 'failed_trials.jsonl', failures, failure)
                self.log(f'Round {round_index} {trial_id} invalid: {exc}')
                continue
            trial_dir = dirs['trials_dir'] / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(trial_dir / 'proposal.json', proposal)
            self._write_json(trial_dir / 'resolved_config.json', resolved['normalized_config'])
            result = execute_family_aware_trial(
                python_executable=self.python_executable,
                fixed_config=locked_search_space.fixed_config,
                proposal_config=resolved['proposal_config'],
                validator_family_spec=locked_family_spec,
                trial_dir=trial_dir,
            )
            result['proposal_id'] = trial_id
            result['hypothesis'] = resolved['hypothesis']
            result['config'] = resolved['proposal_config']
            result['config_hash'] = resolved['config_hash']
            self._write_json(trial_dir / 'result.json', result)
            self._write_json(trial_dir / 'trial_metric_summary.json', build_trial_metric_summary(result))
            if result.get('status') != 'ok':
                execution_failure_count += 1
                no_improve_rounds += 1
                failure = {
                    'trial_id': trial_id,
                    'status': 'failed',
                    'config': resolved['proposal_config'],
                    'config_hash': resolved['config_hash'],
                    'fail_reason': result.get('fail_reason'),
                    'error_type': result.get('error_type'),
                    'error_message': result.get('error_message'),
                }
                self._record_failure(dirs['round_dir'] / 'failed_trials.jsonl', failures, failure)
                self.log(f"Round {round_index} {trial_id} failed: {result.get('fail_reason')}")
                continue
            trial_record = {
                'trial_id': trial_id,
                'status': 'ok',
                'config': resolved['proposal_config'],
                'config_hash': resolved['config_hash'],
                'selection_score': result.get('selection_score'),
                'split_mean': result.get('split_mean'),
                'split_worst': result.get('split_worst'),
                'test_score': result.get('test_score'),
                'test_split_mean': result.get('test_split_mean'),
                'test_split_worst': result.get('test_split_worst'),
                'hypothesis': resolved['hypothesis'],
                'round_index': round_index,
            }
            history.append(trial_record)
            self._write_jsonl(dirs['round_dir'] / 'history.jsonl', history)
            forbidden_hashes.add(resolved['config_hash'])
            improved = best_trial is None or float(result['selection_score']) > float(best_trial['selection_score'])
            if improved:
                best_trial = dict(trial_record)
                no_improve_rounds = 0
                self.log(f"Round {round_index} {trial_id} improved J_trial to {float(result['selection_score']):.4f}")
            else:
                no_improve_rounds += 1
                self.log(f"Round {round_index} {trial_id} completed without improvement: J_trial={float(result['selection_score']):.4f}")
            self._write_json(dirs['round_dir'] / 'best_trial.json', best_trial or {})
        return {
            'history': history,
            'failures': failures,
            'invalid_count': invalid_count,
            'execution_failure_count': execution_failure_count,
            'best_trial': best_trial,
            'stop_reason': stop_reason or 'completed',
        }

    def _write_round_offline_reports(
        self,
        *,
        dirs: Mapping[str, Path],
        round_index: int,
        best_trial: Mapping[str, Any] | None,
        history: List[Dict[str, Any]],
        invalid_count: int,
        execution_failure_count: int,
        warmup_budget: Mapping[str, Any],
        family_aware_budget: Mapping[str, Any],
    ) -> Dict[str, Any]:
        proxy_quality = compute_proxy_quality_summary(history)
        benchmark_summary = compute_benchmark_summary(
            selected_trial=best_trial,
            total_trials=len(history) + invalid_count + execution_failure_count,
            proposal_invalid_count=invalid_count,
            execution_failure_count=execution_failure_count,
            run_completed=True,
            warmup_budget=warmup_budget,
            family_aware_budget=family_aware_budget,
            metadata={
                'project_name': 'AutoValiSearch',
                'validator_protocol': 'validator_family',
                'round_index': round_index,
            },
        )
        self._write_json(dirs['round_dir'] / 'proxy_quality_summary.json', proxy_quality)
        self._write_json(dirs['round_dir'] / 'benchmark_summary.json', benchmark_summary)
        return {'proxy_quality_summary': proxy_quality, 'benchmark_summary': benchmark_summary}

    def _write_project_reports(self) -> Dict[str, Any]:
        table = build_family_summary_table(self.family_records)
        leaderboard = build_family_leaderboard(self.family_records)
        best_family = leaderboard['rows'][0] if leaderboard['rows'] else None
        budget_table = {
            'warmup_paired_probe_budget_by_round': [record.get('warmup_paired_probe_budget') for record in self.family_records],
            'family_aware_train_search_budget_by_round': [record.get('family_aware_train_search_budget') for record in self.family_records],
        }
        final_memory_view = build_family_memory_view(
            self.family_records,
            top_k=self.family_memory_top_k,
            failure_k=self.family_memory_failure_k,
            current_round_index=len(self.family_records) + 1,
        )
        self._write_json(self.workdir / 'family_memory_view.json', final_memory_view)
        project_summary = {
            'status': 'ok',
            'project_name': 'AutoValiSearch',
            'stage': 'Stage 2',
            'num_rounds_completed': len(self.family_records),
            'selected_family_round_index': best_family.get('round_index') if best_family else None,
            'selected_family_hash': best_family.get('family_hash') if best_family else None,
            'proxy_alignment': best_family.get('proxy_alignment') if best_family else None,
            'u_family_online': best_family.get('u_family_online') if best_family else None,
            'j_system': best_family.get('j_system') if best_family else None,
            'best_j_trial': best_family.get('best_j_trial') if best_family else None,
            'budget_table': budget_table,
            'family_memory_view': final_memory_view,
        }
        self._write_json(self.workdir / 'project_offline_summary.json', project_summary)
        self._write_json(self.workdir / 'budget_table.json', budget_table)
        self._write_json(self.workdir / 'family_online_offline_alignment_table.json', table)
        return {'project_summary': project_summary, 'leaderboard': leaderboard, 'table': table}

    def _write_fixed_stable_reference(self) -> None:
        stable_reference = {
            'project_name': 'AutoValiSearch',
            'reference_name': 'fixed_rdds_swa_reference',
            'disturb_mode': 'rsw',
            'bn_mode': self.fixed_config.get('bn_mode'),
            'budget': self.fixed_config.get('budget'),
            'amp': self.fixed_config.get('amp'),
            'rsw_min': self.fixed_config.get('rsw_min'),
            'warmup_probe_count': self.fixed_config.get('warmup_probe_count'),
        }
        self._write_json(self.workdir / 'fixed_stable_reference_config.json', stable_reference)


    def run(self) -> Dict[str, Any]:
        self.log('Starting AutoValiSearch Stage 2 controller')
        self.log(f'Fixed config: {json.dumps(self.fixed_config, ensure_ascii=False, sort_keys=True)}')
        self.log(f'Stop config: {json.dumps(asdict(self.stop_config), ensure_ascii=False, sort_keys=True)}')
        self._write_fixed_stable_reference()
        num_rounds = self._effective_round_count()
        for round_index in range(1, num_rounds + 1):
            self.log(f'Starting family round {round_index}/{num_rounds}')
            dirs = self._prepare_round_dirs(round_index)
            warmup_records, warmup_summary = self._run_warmup(round_index=round_index, dirs=dirs)
            family_memory_view = self._build_round_family_memory_view(round_index, dirs)
            validator_status = self._lock_family(
                round_index=round_index,
                dirs=dirs,
                warmup_summary=warmup_summary,
                family_memory_view=family_memory_view,
            )
            locked_family_spec = dict(validator_status.get('validator_family_spec') or self.fixed_config.get('validator_family_spec') or {})
            locked_family_recipe = dict(validator_status.get('validator_family_recipe') or {})
            family_utility_summary = compute_family_utility_online(
                warmup_records,
                validator_family_spec=locked_family_spec,
                gamma=float(self.fixed_config.get('u_family_gamma', 0.5)),
            )
            self._write_json(dirs['round_dir'] / 'u_family_online_summary.json', family_utility_summary)
            family_search_state = self._run_family_search(
                round_index=round_index,
                dirs=dirs,
                locked_family_spec=locked_family_spec,
                family_utility_summary=family_utility_summary,
            )
            warmup_budget = self._warmup_budget_summary(warmup_records)
            family_aware_budget = self._family_budget_summary(
                family_search_state['history'],
                family_search_state['failures'],
                family_search_state['invalid_count'],
                family_search_state['execution_failure_count'],
            )
            offline_reports = self._write_round_offline_reports(
                dirs=dirs,
                round_index=round_index,
                best_trial=family_search_state['best_trial'],
                history=family_search_state['history'],
                invalid_count=family_search_state['invalid_count'],
                execution_failure_count=family_search_state['execution_failure_count'],
                warmup_budget=warmup_budget,
                family_aware_budget=family_aware_budget,
            )
            round_record = build_family_round_record(
                round_index=round_index,
                family_source=str(validator_status.get('source') or 'unknown'),
                validator_family_recipe=locked_family_recipe,
                validator_family_spec=locked_family_spec,
                warmup_summary=warmup_summary,
                family_utility_summary=family_utility_summary,
                best_trial=family_search_state['best_trial'],
                proxy_quality_summary=offline_reports['proxy_quality_summary'],
                benchmark_summary=offline_reports['benchmark_summary'],
                warmup_budget=warmup_budget,
                family_aware_budget=family_aware_budget,
                invalid_count=family_search_state['invalid_count'],
                execution_failure_count=family_search_state['execution_failure_count'],
            )
            self._write_json(dirs['round_dir'] / 'family_round_summary.json', round_record)
            self.family_records.append(round_record)
            write_family_memory_artifacts(workdir=self.workdir, family_records=self.family_records, current_memory_view=family_memory_view)
            if self.design_only:
                break
            if validator_status.get('status') != 'ok':
                break
        project_reports = self._write_project_reports()
        leaderboard_rows = project_reports['leaderboard']['rows']
        best_family = leaderboard_rows[0] if leaderboard_rows else None
        summary = {
            'status': 'ok',
            'project_name': 'AutoValiSearch',
            'stage': 'Stage 2',
            'num_rounds_completed': len(self.family_records),
            'selected_family_round_index': best_family.get('round_index') if best_family else None,
            'selected_family_hash': best_family.get('family_hash') if best_family else None,
            'proxy_alignment': best_family.get('proxy_alignment') if best_family else None,
            'u_family_online': best_family.get('u_family_online') if best_family else None,
            'j_system': best_family.get('j_system') if best_family else None,
            'best_j_trial': best_family.get('best_j_trial') if best_family else None,
            'stop_reason': 'design_only' if self.design_only else 'completed',
            'warmup_paired_probe_budget': project_reports['project_summary']['budget_table']['warmup_paired_probe_budget_by_round'],
            'family_aware_train_search_budget': project_reports['project_summary']['budget_table']['family_aware_train_search_budget_by_round'],
        }
        self._write_json(self.workdir / 'controller_summary.json', summary)
        return summary

