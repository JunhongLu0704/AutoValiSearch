from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from outer.aggregate_eval import aggregate_train_eval_results, build_eval_plan
from training.validation_protocols import build_warmup_probe_validator_spec

ROOT = Path(__file__).resolve().parents[1]
RUN_TRIAL_SCRIPT = ROOT / 'scripts' / 'run_trial.py'


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')


def _per_split_parallel_evals(fixed_config: Mapping[str, Any]) -> int:
    return max(int(fixed_config.get('per_split_parallel_evals', 2)), 1)


def run_child_trial(*, python_executable: str, trial_config: Mapping[str, Any], trial_dir: Path) -> Dict[str, Any]:
    config_path = trial_dir / 'config.json'
    _write_json(config_path, trial_config)
    command = [python_executable, str(RUN_TRIAL_SCRIPT), '--config', str(config_path), '--trial_dir', str(trial_dir)]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    (trial_dir / 'runner_stdout.txt').write_text(completed.stdout or '', encoding='utf-8')
    (trial_dir / 'runner_stderr.txt').write_text(completed.stderr or '', encoding='utf-8')
    result_path = trial_dir / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
    else:
        result = {
            'status': 'fail',
            'fail_reason': 'missing_result_json',
            'error_type': 'MissingResultJson',
            'error_message': f'run_trial.py exited with code {completed.returncode} without result.json',
        }
        _write_json(result_path, result)
    result['_runner_returncode'] = completed.returncode
    result['_command'] = command
    return result


def _run_child_jobs(job_specs: List[Dict[str, Any]], *, python_executable: str, max_parallel: int) -> List[Dict[str, Any]]:
    if not job_specs:
        return []
    ordered: List[Dict[str, Any] | None] = [None] * len(job_specs)
    if max_parallel <= 1 or len(job_specs) == 1:
        for index, job in enumerate(job_specs):
            ordered[index] = run_child_trial(
                python_executable=python_executable,
                trial_config=job['trial_config'],
                trial_dir=job['trial_dir'],
            )
        return [item for item in ordered if item is not None]
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(job_specs))) as pool:
        future_to_index = {
            pool.submit(
                run_child_trial,
                python_executable=python_executable,
                trial_config=job['trial_config'],
                trial_dir=job['trial_dir'],
            ): index
            for index, job in enumerate(job_specs)
        }
        for future in as_completed(future_to_index):
            ordered[future_to_index[future]] = future.result()
    return [item for item in ordered if item is not None]


def execute_family_aware_trial(
    *,
    python_executable: str,
    fixed_config: Mapping[str, Any],
    proposal_config: Mapping[str, Any],
    validator_family_spec: Mapping[str, Any],
    trial_dir: Path,
) -> Dict[str, Any]:
    raw_config = dict(fixed_config)
    raw_config.update(dict(proposal_config))
    raw_config['validator_protocol'] = 'validator_family'
    raw_config['validator_family_spec'] = dict(validator_family_spec)
    raw_config['validator_spec'] = None
    eval_root = trial_dir / 'aggregate_evals'
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_plan = build_eval_plan(fixed_config)
    eval_records: List[Dict[str, Any]] = []
    split_order: List[str] = []
    evals_by_split: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}
    for index, eval_spec in enumerate(eval_plan, start=1):
        split_name = str(eval_spec['split_name'])
        if split_name not in evals_by_split:
            split_order.append(split_name)
            evals_by_split[split_name] = []
        evals_by_split[split_name].append((index, eval_spec))
    max_parallel = _per_split_parallel_evals(fixed_config)
    for split_name in split_order:
        split_jobs: List[Dict[str, Any]] = []
        for index, eval_spec in evals_by_split[split_name]:
            child_dir = eval_root / f"eval_{index:02d}_{eval_spec['split_name']}_seed_{eval_spec['seed']}"
            child_dir.mkdir(parents=True, exist_ok=True)
            child_config = dict(raw_config)
            child_config['split_dir'] = eval_spec['split_dir']
            child_config['seed'] = int(eval_spec['seed'])
            child_config.pop('eval_split_dirs', None)
            child_config.pop('eval_seeds', None)
            child_config.pop('aggregate_objective', None)
            split_jobs.append(
                {
                    'eval_index': index,
                    'eval_spec': eval_spec,
                    'trial_dir': child_dir,
                    'trial_config': child_config,
                }
            )
        split_results = _run_child_jobs(split_jobs, python_executable=python_executable, max_parallel=max_parallel)
        for job, result in zip(split_jobs, split_results):
            eval_spec = job['eval_spec']
            child_dir = job['trial_dir']
            eval_records.append(
                {
                    'eval_index': job['eval_index'],
                    'eval_key': eval_spec['eval_key'],
                    'split_dir': eval_spec['split_dir'],
                    'split_name': eval_spec['split_name'],
                    'seed': int(eval_spec['seed']),
                    'status': result.get('status'),
                    'selection_score': result.get('selection_score'),
                    'vs_acc': result.get('vs_acc'),
                    'va_avg_acc': result.get('va_avg_acc'),
                    'va_worst_group_acc': result.get('va_worst_group_acc'),
                    'best_test_acc1': result.get('best_test_acc1'),
                    'best_val_epoch': result.get('best_val_epoch'),
                    'validator_member_scores': result.get('validator_member_scores'),
                    'va_family_mean': result.get('va_family_mean'),
                    'va_family_min': result.get('va_family_min'),
                    'va_family_std': result.get('va_family_std'),
                    'va_family_max': result.get('va_family_max'),
                    'fail_reason': result.get('fail_reason'),
                    'error_type': result.get('error_type'),
                    'error_message': result.get('error_message'),
                    'runtime_sec': result.get('runtime_sec'),
                    'config_hash': result.get('config_hash'),
                    'result_path': str((child_dir / 'result.json').relative_to(trial_dir)),
                }
            )
    eval_records.sort(key=lambda item: int(item['eval_index']))
    result = aggregate_train_eval_results(eval_records, aggregate_objective='j_trial')
    result['aggregate_eval_plan'] = eval_plan
    result['validator_protocol'] = 'validator_family'
    result['validator_preset'] = fixed_config.get('validator_preset')
    result['validator_family_spec'] = dict(validator_family_spec)
    return result


def _probe_method_config(base: Mapping[str, Any], *, split_dir: str, seed: int, disturb_mode: str, validator_spec: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(base)
    config['split_dir'] = split_dir
    config['seed'] = int(seed)
    config['disturb_mode'] = disturb_mode
    config['validator_protocol'] = 'handcrafted_va'
    config['validator_preset'] = 'warmup_probe_v1'
    config['validator_spec'] = dict(validator_spec)
    config['validator_family_spec'] = None
    config.pop('eval_split_dirs', None)
    config.pop('eval_seeds', None)
    config.pop('aggregate_objective', None)
    return config


def execute_warmup_paired_probe_bundle(
    *,
    python_executable: str,
    fixed_config: Mapping[str, Any],
    probe_config: Mapping[str, Any],
    probe_dir: Path,
) -> List[Dict[str, Any]]:
    validator_spec = build_warmup_probe_validator_spec()
    records: List[Dict[str, Any]] = []
    base_config = dict(fixed_config)
    base_config.update(dict(probe_config))
    eval_plan = build_eval_plan(fixed_config)
    split_order: List[str] = []
    evals_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for eval_spec in eval_plan:
        split_name = str(eval_spec['split_name'])
        if split_name not in evals_by_split:
            split_order.append(split_name)
            evals_by_split[split_name] = []
        evals_by_split[split_name].append(eval_spec)
    max_parallel = _per_split_parallel_evals(fixed_config)
    for split_name in split_order:
        split_evals = evals_by_split[split_name]
        for method_name, disturb_mode in [('erm', 'none'), ('stable', 'rsw')]:
            method_jobs: List[Dict[str, Any]] = []
            for eval_spec in split_evals:
                child_dir = probe_dir / f"{eval_spec['split_name']}_seed_{eval_spec['seed']}_{method_name}"
                child_dir.mkdir(parents=True, exist_ok=True)
                child_config = _probe_method_config(
                    base_config,
                    split_dir=eval_spec['split_dir'],
                    seed=int(eval_spec['seed']),
                    disturb_mode=disturb_mode,
                    validator_spec=validator_spec,
                )
                method_jobs.append(
                    {
                        'eval_spec': eval_spec,
                        'method_name': method_name,
                        'trial_dir': child_dir,
                        'trial_config': child_config,
                    }
                )
            method_results = _run_child_jobs(method_jobs, python_executable=python_executable, max_parallel=max_parallel)
            for job, result in zip(method_jobs, method_results):
                eval_spec = job['eval_spec']
                child_dir = job['trial_dir']
                records.append(
                    {
                        'probe_config': dict(probe_config),
                        'split_dir': eval_spec['split_dir'],
                        'split_name': eval_spec['split_name'],
                        'seed': int(eval_spec['seed']),
                        'method': job['method_name'],
                        'status': result.get('status'),
                        'selection_score': result.get('selection_score'),
                        'vs_acc': result.get('vs_acc'),
                        'va_group_acc': dict(result.get('va_group_acc') or {}),
                        'best_test_acc1': result.get('best_test_acc1'),
                        'fail_reason': result.get('fail_reason'),
                        'error_type': result.get('error_type'),
                        'error_message': result.get('error_message'),
                        'runtime_sec': result.get('runtime_sec'),
                        'result_path': str((child_dir / 'result.json').relative_to(probe_dir)),
                    }
                )
    return records

