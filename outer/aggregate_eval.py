from __future__ import annotations

import math
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from training.validation_protocols import aggregate_validator_family_metrics, aggregate_validator_metrics, harmonic_mean

VALID_AGGREGATE_OBJECTIVES = {'j_trial'}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def normalize_aggregate_objective(value: Any) -> str:
    normalized = str(value or 'j_trial').lower()
    if normalized != 'j_trial':
        raise ValueError(f'Unsupported aggregate_objective: {value}')
    return normalized


def build_eval_plan(fixed_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for split_dir in fixed_config.get('eval_split_dirs', [fixed_config['split_dir']]):
        split_name = Path(str(split_dir)).name
        for seed in fixed_config.get('eval_seeds', [fixed_config['seed']]):
            plan.append(
                {
                    'split_dir': str(split_dir),
                    'split_name': split_name,
                    'seed': int(seed),
                    'eval_key': f'{split_name}__seed_{int(seed)}',
                }
            )
    return plan


def build_trial_metric_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        'status': result.get('status'),
        'selection_metric_name': result.get('selection_metric_name'),
        'selection_score': result.get('selection_score'),
        'j_trial': result.get('selection_score'),
        'split_mean': result.get('split_mean'),
        'split_worst': result.get('split_worst'),
        'seed_mean_scores_by_split': result.get('seed_mean_scores_by_split'),
        'child_eval_scores': result.get('child_eval_scores'),
        'num_evals': result.get('num_evals'),
        'num_successful_evals': result.get('num_successful_evals'),
        'num_failed_evals': result.get('num_failed_evals'),
        'validator_protocol': result.get('validator_protocol'),
        'validator_preset': result.get('validator_preset'),
        'test_score': result.get('test_score'),
        'test_split_mean': result.get('test_split_mean'),
        'test_split_worst': result.get('test_split_worst'),
        'test_seed_mean_scores_by_split': result.get('test_seed_mean_scores_by_split'),
        'best_test_acc1_mean': result.get('best_test_acc1_mean'),
        'best_test_acc1_worst': result.get('best_test_acc1_worst'),
        'failed_eval_keys': result.get('failed_eval_keys'),
        'fail_reason': result.get('fail_reason'),
        'error_type': result.get('error_type'),
        'error_message': result.get('error_message'),
    }


def _group_scores_by_split(records: Sequence[Mapping[str, Any]], field: str) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for record in records:
        score = _safe_float(record.get(field))
        split_name = str(record.get('split_name') or Path(str(record.get('split_dir', 'unknown_split'))).name)
        if score is None:
            continue
        grouped.setdefault(split_name, []).append(float(score))
    return grouped


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _mean_metric(records: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [_safe_float(record.get(field)) for record in records]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return _mean(clean)


def aggregate_train_eval_results(
    eval_records: Iterable[Mapping[str, Any]],
    *,
    aggregate_objective: str = 'j_trial',
) -> Dict[str, Any]:
    aggregate_objective = normalize_aggregate_objective(aggregate_objective)
    eval_list = [dict(record) for record in eval_records]
    success_records = [
        record for record in eval_list if record.get('status') == 'ok' and _safe_float(record.get('selection_score')) is not None
    ]
    failed_records = [record for record in eval_list if record not in success_records]
    payload: Dict[str, Any] = {
        'status': 'ok' if not failed_records and success_records else 'fail',
        'aggregate_objective': aggregate_objective,
        'eval_results': eval_list,
        'num_evals': len(eval_list),
        'num_successful_evals': len(success_records),
        'num_failed_evals': len(failed_records),
        'failed_eval_keys': [record.get('eval_key') for record in failed_records],
        'selection_metric_name': 'J_trial',
    }
    if not success_records:
        payload.update(
            {
                'fail_reason': 'all_aggregate_evals_failed',
                'error_type': 'AggregateEvalFailure',
                'error_message': 'No successful evals were available for aggregate scoring.',
            }
        )
        return payload
    if failed_records:
        payload.update(
            {
                'status': 'fail',
                'fail_reason': 'partial_aggregate_eval_failure',
                'error_type': 'AggregateEvalFailure',
                'error_message': f'{len(failed_records)} / {len(eval_list)} aggregate evals failed.',
            }
        )
        return payload
    child_scores = [float(record['selection_score']) for record in success_records]
    child_eval_scores = {str(record.get('eval_key')): round(float(record['selection_score']), 6) for record in success_records}
    split_scores = _group_scores_by_split(success_records, 'selection_score')
    seed_mean_scores_by_split = {split: round(_mean(values), 6) for split, values in split_scores.items() if values}
    split_values = list(seed_mean_scores_by_split.values())
    split_mean = _mean(split_values)
    split_worst = min(split_values)
    j_trial = harmonic_mean(split_mean, split_worst)
    test_split_groups = _group_scores_by_split(success_records, 'best_test_acc1')
    test_seed_mean_scores_by_split = {split: round(_mean(values), 6) for split, values in test_split_groups.items() if values}
    test_split_values = list(test_seed_mean_scores_by_split.values())
    test_split_mean = _mean(test_split_values) if test_split_values else None
    test_split_worst = min(test_split_values) if test_split_values else None
    test_score = harmonic_mean(test_split_mean, test_split_worst) if test_split_values else None
    member_score_summary = {}
    for record in success_records:
        member_scores = record.get('validator_member_scores') or {}
        if isinstance(member_scores, Mapping):
            for name, value in member_scores.items():
                score = _safe_float(value)
                if score is None:
                    continue
                member_score_summary.setdefault(str(name), []).append(score)
    test_values = [float(record['best_test_acc1']) for record in success_records if _safe_float(record.get('best_test_acc1')) is not None]
    payload.update(
        {
            'selection_score': round(j_trial, 6),
            'j_trial': round(j_trial, 6),
            'split_mean': round(split_mean, 6),
            'split_worst': round(split_worst, 6),
            'seed_mean_scores_by_split': seed_mean_scores_by_split,
            'child_eval_scores': child_eval_scores,
            'selection_score_std': round(pstdev(child_scores) if len(child_scores) > 1 else 0.0, 6),
            'vs_acc_mean': _mean_metric(success_records, 'vs_acc'),
            'va_family_mean_mean': _mean_metric(success_records, 'va_family_mean'),
            'va_family_min_mean': _mean_metric(success_records, 'va_family_min'),
            'va_family_std_mean': _mean_metric(success_records, 'va_family_std'),
            'validator_member_scores_mean': {name: round(_mean(values), 6) for name, values in member_score_summary.items()},
            'best_test_acc1_mean': _mean_metric(success_records, 'best_test_acc1'),
            'best_test_acc1_worst': min(test_values) if test_values else None,
            'best_val_epoch_mean': _mean_metric(success_records, 'best_val_epoch'),
            'test_seed_mean_scores_by_split': test_seed_mean_scores_by_split,
            'test_split_mean': None if test_split_mean is None else round(test_split_mean, 6),
            'test_split_worst': None if test_split_worst is None else round(test_split_worst, 6),
            'test_score': None if test_score is None else round(test_score, 6),
        }
    )
    return payload


def _average_rank(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + end + 2) / 2.0
        for index in range(start, end + 1):
            ranks[indexed[index][0]] = avg_rank
        start = end + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    numerator = sum(lhs * rhs for lhs, rhs in zip(centered_x, centered_y))
    return float(numerator / (denom_x * denom_y))


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_rank(xs), _average_rank(ys))


def compute_proxy_quality_summary(trial_pool: Iterable[Mapping[str, Any]], *, top_k: int = 3) -> Dict[str, Any]:
    valid_trials = []
    for record in trial_pool:
        score = _safe_float(record.get('selection_score'))
        test_score = _safe_float(record.get('test_score'))
        if score is None or test_score is None:
            continue
        valid_trials.append({'trial_id': record.get('trial_id'), 'selection_score': float(score), 'test_score': float(test_score)})
    payload: Dict[str, Any] = {
        'status': 'ok',
        'num_valid_trials': len(valid_trials),
        'proxy_alignment': None,
        'spearman_j_trial_vs_t': None,
        'top1_regret': None,
        'regret_norm': None,
        'topk_hit_rate': None,
        'top_k': min(top_k, len(valid_trials)) if valid_trials else 0,
        'selected_trial_id': None,
        'oracle_trial_id': None,
    }
    if len(valid_trials) < 2:
        payload.update({'status': 'insufficient_data', 'reason': 'need_at_least_two_valid_trials'})
        return payload
    proxy_sorted = sorted(valid_trials, key=lambda item: item['selection_score'], reverse=True)
    test_sorted = sorted(valid_trials, key=lambda item: item['test_score'], reverse=True)
    selected = proxy_sorted[0]
    oracle = test_sorted[0]
    test_values = [item['test_score'] for item in valid_trials]
    regret = float(oracle['test_score'] - selected['test_score'])
    value_span = max(test_values) - min(test_values)
    regret_norm = 0.0 if value_span <= 0.0 else max(0.0, regret) / value_span
    spearman = spearman_rank_correlation(
        [item['selection_score'] for item in valid_trials],
        [item['test_score'] for item in valid_trials],
    )
    k = min(top_k, len(valid_trials))
    proxy_topk = {item['trial_id'] for item in proxy_sorted[:k]}
    test_topk = {item['trial_id'] for item in test_sorted[:k]}
    topk_hit_rate = len(proxy_topk & test_topk) / float(k)
    proxy_alignment = 0.5 * float(spearman if spearman is not None else 0.0) + 0.5 * (1.0 - regret_norm)
    payload.update(
        {
            'proxy_alignment': round(proxy_alignment, 6),
            'spearman_j_trial_vs_t': None if spearman is None else round(float(spearman), 6),
            'top1_regret': round(regret, 6),
            'regret_norm': round(regret_norm, 6),
            'topk_hit_rate': round(topk_hit_rate, 6),
            'selected_trial_id': selected.get('trial_id'),
            'oracle_trial_id': oracle.get('trial_id'),
        }
    )
    return payload


def compute_benchmark_summary(
    *,
    selected_trial: Mapping[str, Any] | None,
    total_trials: int,
    proposal_invalid_count: int,
    execution_failure_count: int,
    run_completed: bool,
    warmup_budget: Mapping[str, Any],
    family_aware_budget: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    valid_proposals = max(total_trials - proposal_invalid_count, 0)
    proposal_valid_rate = None if total_trials <= 0 else valid_proposals / float(total_trials)
    execution_failure_rate = None if total_trials <= 0 else execution_failure_count / float(total_trials)
    test_score = _safe_float((selected_trial or {}).get('test_score'))
    summary = {
        'status': 'ok' if run_completed else 'incomplete',
        'j_system': None if test_score is None else round(float(test_score), 6),
        'score_at_budget': None if test_score is None else round(float(test_score), 6),
        'proposal_valid_rate': None if proposal_valid_rate is None else round(proposal_valid_rate, 6),
        'execution_failure_rate': None if execution_failure_rate is None else round(execution_failure_rate, 6),
        'run_completion_rate': 1.0 if run_completed else 0.0,
        'selected_trial_id': (selected_trial or {}).get('trial_id'),
        'selected_selection_score': (selected_trial or {}).get('selection_score'),
        'selected_split_mean': (selected_trial or {}).get('split_mean'),
        'selected_split_worst': (selected_trial or {}).get('split_worst'),
        'selected_test_split_mean': (selected_trial or {}).get('test_split_mean'),
        'selected_test_split_worst': (selected_trial or {}).get('test_split_worst'),
        'total_trials': total_trials,
        'proposal_invalid_count': proposal_invalid_count,
        'execution_failure_count': execution_failure_count,
        'warmup_paired_probe_budget': dict(warmup_budget),
        'family_aware_train_search_budget': dict(family_aware_budget),
    }
    summary.update(metadata)
    return summary


def _member_scores_from_probe_stats(
    method_vs_acc: float,
    method_group_scores: Mapping[str, float],
    validator_family_spec: Mapping[str, Any],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for member in validator_family_spec.get('validators', []):
        group_scores = {}
        for group in (member.get('spec') or {}).get('groups', []):
            group_name = str(group.get('name'))
            if group_name not in method_group_scores:
                raise ValueError(f'Missing warmup probe score for group {group_name}')
            group_scores[group_name] = float(method_group_scores[group_name])
        member_metrics = aggregate_validator_metrics(member['protocol'], vs_acc=method_vs_acc, va_group_acc=group_scores)
        scores[str(member['name'])] = float(member_metrics['selection_score'])
    return scores


def compute_family_utility_online(
    paired_records: Iterable[Mapping[str, Any]],
    *,
    validator_family_spec: Mapping[str, Any],
    gamma: float = 0.5,
) -> Dict[str, Any]:
    pair_index: Dict[tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for record in paired_records:
        key = (str(record['split_name']), int(record['seed']))
        pair_index.setdefault(key, {})[str(record['method']).lower()] = dict(record)
    sep_by_split: Dict[str, List[float]] = {}
    stable_mu_advantages: List[float] = []
    stable_var_advantages: List[float] = []
    method_pairs = []
    for (split_name, seed), methods in pair_index.items():
        if 'erm' not in methods or 'stable' not in methods:
            continue
        erm = methods['erm']
        stable = methods['stable']
        if erm.get('status') != 'ok' or stable.get('status') != 'ok':
            continue
        erm_vs = float(erm.get('vs_acc', 0.0))
        stable_vs = float(stable.get('vs_acc', 0.0))
        erm_group_scores = dict((erm.get('va_group_acc') or {}))
        stable_group_scores = dict((stable.get('va_group_acc') or {}))
        erm_member_scores = _member_scores_from_probe_stats(erm_vs, erm_group_scores, validator_family_spec)
        stable_member_scores = _member_scores_from_probe_stats(stable_vs, stable_group_scores, validator_family_spec)
        erm_family = aggregate_validator_family_metrics(validator_family_spec, vs_acc=erm_vs, validator_member_scores=erm_member_scores)
        stable_family = aggregate_validator_family_metrics(validator_family_spec, vs_acc=stable_vs, validator_member_scores=stable_member_scores)
        erm_values = list(erm_member_scores.values())
        stable_values = list(stable_member_scores.values())
        mu_erm = float(sum(erm_values) / len(erm_values))
        mu_stable = float(sum(stable_values) / len(stable_values))
        sigma_erm = float(pstdev(erm_values) if len(erm_values) > 1 else 0.0)
        sigma_stable = float(pstdev(stable_values) if len(stable_values) > 1 else 0.0)
        sep_sr = (mu_stable - mu_erm) + float(gamma) * (sigma_erm - sigma_stable)
        sep_by_split.setdefault(split_name, []).append(sep_sr)
        stable_mu_advantages.append(mu_stable - mu_erm)
        stable_var_advantages.append(sigma_erm - sigma_stable)
        method_pairs.append(
            {
                'split_name': split_name,
                'seed': seed,
                'sep_sr': round(sep_sr, 6),
                'mu_stable': round(mu_stable, 6),
                'mu_erm': round(mu_erm, 6),
                'sigma_stable': round(sigma_stable, 6),
                'sigma_erm': round(sigma_erm, 6),
                'stable_member_scores': {k: round(v, 6) for k, v in stable_member_scores.items()},
                'erm_member_scores': {k: round(v, 6) for k, v in erm_member_scores.items()},
                'stable_family_score': stable_family['selection_score'],
                'erm_family_score': erm_family['selection_score'],
            }
        )
    if not sep_by_split:
        return {
            'status': 'insufficient_data',
            'u_family_online': None,
            'family_seed_mean_sep_by_split': {},
            'family_split_mean_sep': None,
            'family_split_worst_sep': None,
            'stable_mean_advantage': None,
            'stable_variance_advantage': None,
            'method_pairs': method_pairs,
        }
    split_seed_means = {split: _mean(values) for split, values in sep_by_split.items() if values}
    split_values = list(split_seed_means.values())
    split_mean = _mean(split_values)
    split_worst = min(split_values)
    u_family_online = harmonic_mean(split_mean, split_worst)
    return {
        'status': 'ok',
        'u_family_online': round(u_family_online, 6),
        'family_seed_mean_sep_by_split': {split: round(value, 6) for split, value in split_seed_means.items()},
        'family_split_mean_sep': round(split_mean, 6),
        'family_split_worst_sep': round(split_worst, 6),
        'stable_mean_advantage': round(_mean(stable_mu_advantages), 6) if stable_mu_advantages else None,
        'stable_variance_advantage': round(_mean(stable_var_advantages), 6) if stable_var_advantages else None,
        'method_pairs': method_pairs,
    }


def build_warmup_summary(paired_records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    status_counts = {'ok': 0, 'fail': 0}
    for record in paired_records:
        method = str(record.get('method')).lower()
        status = str(record.get('status'))
        status_counts['ok' if status == 'ok' else 'fail'] += 1
        if status != 'ok':
            continue
        for group_name, score in dict(record.get('va_group_acc') or {}).items():
            grouped.setdefault(group_name, {'erm': [], 'stable': []}).setdefault(method, []).append(float(score))
    group_advantages = {}
    for group_name, methods in grouped.items():
        stable_scores = methods.get('stable') or []
        erm_scores = methods.get('erm') or []
        if not stable_scores or not erm_scores:
            continue
        group_advantages[group_name] = {
            'stable_mean': round(_mean(stable_scores), 6),
            'erm_mean': round(_mean(erm_scores), 6),
            'stable_advantage': round(_mean(stable_scores) - _mean(erm_scores), 6),
        }
    return {
        'status': 'ok' if status_counts['ok'] > 0 else 'insufficient_data',
        'num_records': sum(status_counts.values()),
        'num_successful_records': status_counts['ok'],
        'num_failed_records': status_counts['fail'],
        'group_advantages': group_advantages,
    }

