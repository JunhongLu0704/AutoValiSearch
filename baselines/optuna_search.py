from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outer.aggregate_eval import build_trial_metric_summary, compute_benchmark_summary, compute_proxy_quality_summary
from outer.execution import execute_family_aware_trial
from outer.schema import SearchSpace, compute_candidate_hash, default_search_space

DEFAULT_PYTHON_EXE = os.environ.get('VARIEDSTABLENET_PYTHON', r'C:\Users\123\.conda\envs\OOD\python.exe')
SUPPORTED_OPTUNA_SAMPLERS = {'random', 'tpe', 'botorch'}


class OptunaDependencyError(RuntimeError):
    pass


class BaselineTrialError(RuntimeError):
    def __init__(self, message: str, *, result: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.result = result or {}


@dataclass
class OptunaStopConfig:
    n_trials: int = 20
    timeout_sec: Optional[float] = None


@dataclass
class SearchPolicyConfig:
    search_policy_preset: str = 'search_cold_tpe'
    warm_start_rule: str = 'cold_start'
    exploration_ratio: float = 0.30
    inner_proposer_mode: str = 'tpe'
    restart_rule: str = 'no_restart'
    diversification_rule: str = 'none'
    warm_start_pool: Optional[List[Dict[str, Any]]] = None
    warm_start_source_summary: Optional[Dict[str, Any]] = None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')


def _write_jsonl(path: Path, records: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def _load_optuna():
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError as exc:
        raise OptunaDependencyError('Optuna is not installed. Install it with `python -m pip install -U optuna`.') from exc
    return optuna


def _build_sampler(optuna_module: Any, sampler_name: str, seed: int):
    sampler_name = str(sampler_name).lower()
    if sampler_name not in SUPPORTED_OPTUNA_SAMPLERS:
        raise ValueError(f'Unsupported Optuna sampler: {sampler_name}')
    if sampler_name == 'random':
        return optuna_module.samplers.RandomSampler(seed=seed)
    if sampler_name == 'tpe':
        return optuna_module.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    try:
        from optuna.integration import BoTorchSampler  # type: ignore
    except Exception as exc:
        raise OptunaDependencyError(
            'BoTorchSampler requires Optuna integration support and BoTorch. Install with `python -m pip install -U optuna optuna-integration botorch gpytorch`.'
        ) from exc
    return BoTorchSampler(seed=seed)


def _canonicalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'lr': float(config['lr']),
        'lambdap': float(config['lambdap']),
        'epochp': int(config['epochp']),
        'weight_decay': float(config['weight_decay']),
    }


def _config_key(config: Dict[str, Any]) -> str:
    return json.dumps(_canonicalize_config(config), ensure_ascii=False, sort_keys=True)


def _suggest_config(trial: Any, search_space: SearchSpace) -> Dict[str, Any]:
    return {
        'lr': trial.suggest_categorical('lr', list(search_space.tunables['lr'])),
        'lambdap': trial.suggest_categorical('lambdap', list(search_space.tunables['lambdap'])),
        'epochp': trial.suggest_categorical('epochp', list(search_space.tunables['epochp'])),
        'weight_decay': trial.suggest_categorical('weight_decay', list(search_space.tunables['weight_decay'])),
    }


def _sample_random_config(search_space: SearchSpace, rng: random.Random) -> Dict[str, Any]:
    return {field: rng.choice(list(values)) for field, values in search_space.tunables.items()}


def _anchor_configs(search_space: SearchSpace) -> List[Dict[str, Any]]:
    defaults = [
        {'lr': 0.001, 'lambdap': 2.0, 'epochp': 1, 'weight_decay': 0.0},
        {'lr': 0.001, 'lambdap': 4.0, 'epochp': 4, 'weight_decay': 0.0001},
        {'lr': 0.003, 'lambdap': 2.0, 'epochp': 2, 'weight_decay': 0.0001},
        {'lr': 0.003, 'lambdap': 8.0, 'epochp': 2, 'weight_decay': 0.0005},
    ]
    anchored: List[Dict[str, Any]] = []
    for candidate in defaults:
        normalized = {}
        for field, preferred in candidate.items():
            values = list(search_space.tunables[field])
            if preferred in values:
                normalized[field] = preferred
            else:
                normalized[field] = min(values, key=lambda value: abs(float(value) - float(preferred)))
        anchored.append(_canonicalize_config(normalized))
    return anchored


def _build_exploration_configs(search_space: SearchSpace, count: int, seed: int, forbidden: set[str]) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []
    for anchor in _anchor_configs(search_space):
        key = _config_key(anchor)
        if key in forbidden:
            continue
        forbidden.add(key)
        results.append(anchor)
        if len(results) >= count:
            return results
    attempts = 0
    while len(results) < count and attempts < 512:
        attempts += 1
        config = _canonicalize_config(_sample_random_config(search_space, rng))
        key = _config_key(config)
        if key in forbidden:
            continue
        forbidden.add(key)
        results.append(config)
    return results


def run_optuna_search(
    *,
    workdir: str,
    fixed_config: Dict[str, Any],
    sampler_name: str,
    stop_config: OptunaStopConfig,
    python_executable: str = DEFAULT_PYTHON_EXE,
    study_name: Optional[str] = None,
    search_policy: Optional[SearchPolicyConfig] = None,
) -> Dict[str, Any]:
    optuna = _load_optuna()
    workdir_path = Path(workdir).resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)
    trials_dir = workdir_path / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)
    search_space = default_search_space(fixed_config)
    sampler = _build_sampler(optuna, sampler_name=sampler_name, seed=int(fixed_config['seed']))
    storage = f"sqlite:///{(workdir_path / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(
        study_name=study_name or f'autovalisearch_{sampler_name}_baseline',
        direction='maximize',
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    search_policy = search_policy or SearchPolicyConfig()
    history_records: list[Dict[str, Any]] = []
    started_at = time.time()
    queued_sources: Dict[str, str] = {}
    local_source_count = 0
    exploration_source_count = 0
    restart_triggered = False
    restart_events: List[Dict[str, Any]] = []
    best_value: Optional[float] = None
    no_improve_count = 0

    def enqueue_config(config: Dict[str, Any], source: str) -> None:
        canonical = _canonicalize_config(config)
        key = _config_key(canonical)
        if key in queued_sources:
            return
        study.enqueue_trial(canonical)
        queued_sources[key] = source

    warm_start_pool = [_canonicalize_config(item) for item in list(search_policy.warm_start_pool or [])]
    warm_start_keys = {_config_key(item) for item in warm_start_pool}
    exploration_quota = min(stop_config.n_trials, int(math.ceil(stop_config.n_trials * float(search_policy.exploration_ratio))))
    local_quota = max(stop_config.n_trials - exploration_quota, 0)
    for config in warm_start_pool[:local_quota]:
        enqueue_config(config, 'warm_start_local')
    local_source_count = min(len(warm_start_pool), local_quota)
    forbidden = set(warm_start_keys)
    staged_exploration = _build_exploration_configs(
        search_space,
        count=max(exploration_quota, 1 if float(search_policy.exploration_ratio) > 0 else 0),
        seed=int(fixed_config['seed']) + 13,
        forbidden=forbidden,
    )
    initial_exploration = staged_exploration if search_policy.restart_rule == 'no_restart' else staged_exploration[: max(1, len(staged_exploration) // 2)]
    deferred_exploration = staged_exploration[len(initial_exploration):]
    for config in initial_exploration:
        enqueue_config(config, 'exploration_seed')
    exploration_source_count += len(initial_exploration)

    def objective(trial: Any) -> float:
        trial_index = trial.number + 1
        trial_id = f'trial_{trial_index:04d}'
        trial_dir = trials_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        proposal_config = _suggest_config(trial, search_space)
        result = execute_family_aware_trial(
            python_executable=python_executable,
            fixed_config=search_space.fixed_config,
            proposal_config=proposal_config,
            validator_family_spec=search_space.fixed_config.get('validator_family_spec') or {},
            trial_dir=trial_dir,
        )
        result['proposal_id'] = trial_id
        result['config'] = proposal_config
        result['config_hash'] = compute_candidate_hash(proposal_config, search_space)
        source = queued_sources.get(_config_key(proposal_config), 'sampler_model')
        _write_json(trial_dir / 'proposal.json', {'proposal_id': trial_id, 'config': proposal_config, 'sampler_name': sampler_name, 'proposal_source': source})
        _write_json(trial_dir / 'result.json', result)
        _write_json(trial_dir / 'trial_metric_summary.json', build_trial_metric_summary(result))
        history_records.append(
            {
                'trial_number': trial.number,
                'trial_id': trial_id,
                'sampler_name': sampler_name,
                'proposal_config': proposal_config,
                'proposal_source': source,
                'result': result,
            }
        )
        _write_jsonl(workdir_path / 'history.jsonl', history_records)
        trial.set_user_attr('trial_id', trial_id)
        trial.set_user_attr('proposal_config', proposal_config)
        trial.set_user_attr('proposal_source', source)
        trial.set_user_attr('status', result.get('status'))
        trial.set_user_attr('split_mean', result.get('split_mean'))
        trial.set_user_attr('split_worst', result.get('split_worst'))
        trial.set_user_attr('test_score', result.get('test_score'))
        trial.set_user_attr('test_split_mean', result.get('test_split_mean'))
        trial.set_user_attr('test_split_worst', result.get('test_split_worst'))
        if result.get('status') != 'ok':
            raise BaselineTrialError(f'{trial_id} failed with status={result.get("status")}', result=result)
        return float(result['selection_score'])

    while len(study.trials) < stop_config.n_trials:
        if stop_config.timeout_sec is not None and time.time() - started_at >= stop_config.timeout_sec:
            break
        study.optimize(
            objective,
            n_trials=1,
            timeout=None,
            catch=(BaselineTrialError,),
            gc_after_trial=True,
            show_progress_bar=False,
        )
        current = study.trials[-1]
        if current.state.name == 'COMPLETE' and current.value is not None:
            value = float(current.value)
            if best_value is None or value > best_value:
                best_value = value
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif current.state.name == 'FAIL':
            no_improve_count += 1
        half_trigger = (
            search_policy.restart_rule == 'restart_halfway_once'
            and not restart_triggered
            and len(study.trials) >= max(stop_config.n_trials // 2, 1)
        )
        stagnation_trigger = (
            search_policy.restart_rule == 'restart_if_no_improve_4'
            and not restart_triggered
            and no_improve_count >= 4
        )
        if (half_trigger or stagnation_trigger) and deferred_exploration:
            restart_triggered = True
            for config in deferred_exploration:
                enqueue_config(config, 'restart_exploration')
            exploration_source_count += len(deferred_exploration)
            restart_events.append({
                'trigger': 'halfway' if half_trigger else 'no_improve_4',
                'after_trial_count': len(study.trials),
                'enqueued_count': len(deferred_exploration),
            })
            deferred_exploration = []

    completed = [trial for trial in study.trials if trial.state.name == 'COMPLETE']
    failed = [trial for trial in study.trials if trial.state.name == 'FAIL']
    best_summary: Optional[Dict[str, Any]] = None
    if completed:
        best_trial = study.best_trial
        best_summary = {
            'trial_number': best_trial.number,
            'trial_id': best_trial.user_attrs.get('trial_id'),
            'selection_score': best_trial.value,
            'proposal_config': best_trial.user_attrs.get('proposal_config'),
            'proposal_source': best_trial.user_attrs.get('proposal_source'),
            'split_mean': best_trial.user_attrs.get('split_mean'),
            'split_worst': best_trial.user_attrs.get('split_worst'),
            'test_score': best_trial.user_attrs.get('test_score'),
            'test_split_mean': best_trial.user_attrs.get('test_split_mean'),
            'test_split_worst': best_trial.user_attrs.get('test_split_worst'),
        }
    proxy_quality_summary = compute_proxy_quality_summary(
        [
            {
                'trial_id': record.get('trial_id'),
                'selection_score': (record.get('result') or {}).get('selection_score'),
                'test_score': (record.get('result') or {}).get('test_score'),
            }
            for record in history_records
            if (record.get('result') or {}).get('status') == 'ok'
        ]
    )
    benchmark_summary = compute_benchmark_summary(
        selected_trial=best_summary,
        total_trials=len(study.trials),
        proposal_invalid_count=0,
        execution_failure_count=len(failed),
        run_completed=True,
        warmup_budget={'warmup_probe_count': 0, 'warmup_child_runs': 0, 'warmup_successful_child_runs': 0},
        family_aware_budget={
            'max_trials': stop_config.n_trials,
            'completed_trials': len(completed),
            'terminal_rounds': len(study.trials),
            'proposal_invalid_count': 0,
            'execution_failure_count': len(failed),
        },
        metadata={
            'project_name': 'AutoValiSearch',
            'sampler_name': sampler_name,
            'study_name': study.study_name,
            'search_policy_preset': search_policy.search_policy_preset,
        },
    )
    proposal_source_counts: Dict[str, int] = {}
    for trial in study.trials:
        source = str(trial.user_attrs.get('proposal_source', 'unknown'))
        proposal_source_counts[source] = proposal_source_counts.get(source, 0) + 1
    inner_search_summary = {
        'search_policy_preset': search_policy.search_policy_preset,
        'warm_start_rule': search_policy.warm_start_rule,
        'exploration_ratio': search_policy.exploration_ratio,
        'inner_proposer_mode': search_policy.inner_proposer_mode,
        'restart_rule': search_policy.restart_rule,
        'diversification_rule': search_policy.diversification_rule,
        'warm_start_source_summary': search_policy.warm_start_source_summary or {'warm_start_rule': search_policy.warm_start_rule, 'total_configs': len(warm_start_pool), 'sources': []},
        'warm_start_pool_size': len(warm_start_pool),
        'local_seeded_proposal_count': local_source_count,
        'exploration_seeded_proposal_count': exploration_source_count,
        'restart_triggered': restart_triggered,
        'restart_events': restart_events,
        'proposal_source_counts': proposal_source_counts,
    }
    summary = {
        'status': 'ok',
        'sampler_name': sampler_name,
        'study_name': study.study_name,
        'storage': storage,
        'n_trials_requested': stop_config.n_trials,
        'timeout_sec': stop_config.timeout_sec,
        'num_trials_total': len(study.trials),
        'num_complete': len(completed),
        'num_failed': len(failed),
        'best_trial': best_summary,
        'proxy_alignment': proxy_quality_summary.get('proxy_alignment'),
        'j_system': benchmark_summary.get('j_system'),
        'proposal_valid_rate': benchmark_summary.get('proposal_valid_rate'),
        'execution_failure_rate': benchmark_summary.get('execution_failure_rate'),
        'run_completion_rate': benchmark_summary.get('run_completion_rate'),
        'runtime_sec': round(time.time() - started_at, 4),
        'workdir': str(workdir_path),
        'search_policy_preset': search_policy.search_policy_preset,
        'warm_start_rule': search_policy.warm_start_rule,
        'exploration_ratio': search_policy.exploration_ratio,
        'inner_proposer_mode': search_policy.inner_proposer_mode,
        'restart_rule': search_policy.restart_rule,
        'diversification_rule': search_policy.diversification_rule,
        'warm_start_source_summary': inner_search_summary['warm_start_source_summary'],
        'restart_triggered': restart_triggered,
        'proposal_source_counts': proposal_source_counts,
    }
    _write_json(workdir_path / 'study_summary.json', summary)
    _write_json(workdir_path / 'proxy_quality_summary.json', proxy_quality_summary)
    _write_json(workdir_path / 'benchmark_summary.json', benchmark_summary)
    _write_json(workdir_path / 'inner_search_summary.json', inner_search_summary)
    _write_json(
        workdir_path / 'study_trials.json',
        {
            'trials': [
                {
                    'number': trial.number,
                    'state': trial.state.name,
                    'value': trial.value,
                    'params': dict(trial.params),
                    'user_attrs': dict(trial.user_attrs),
                }
                for trial in study.trials
            ]
        },
    )
    return summary

