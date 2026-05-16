from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def canonical_family_hash(validator_family_spec: Mapping[str, Any]) -> str:
    import hashlib

    payload = json.dumps(dict(validator_family_spec), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def summarize_family_strengths(record: Mapping[str, Any]) -> str:
    proxy = _safe_float(record.get('proxy_alignment'))
    utility = _safe_float(record.get('u_family_online'))
    j_system = _safe_float(record.get('j_system'))
    failed = int(record.get('execution_failure_count') or 0)
    parts: List[str] = []
    if proxy is not None:
        parts.append(f'ProxyAlignment={proxy:.4f}')
    if utility is not None:
        parts.append(f'U_family_online={utility:.4f}')
    if j_system is not None:
        parts.append(f'J_system={j_system:.4f}')
    if failed > 0:
        parts.append(f'execution_failures={failed}')
    if proxy is not None and utility is not None:
        if proxy >= 0.5 and utility >= 0.0:
            parts.append('good online/offline behavior')
        elif proxy < 0.3 and utility >= 0.0:
            parts.append('online signal stronger than offline alignment')
        elif proxy >= 0.5 and utility < 0.0:
            parts.append('offline alignment good despite weak online utility')
    return '; '.join(parts) if parts else 'no summary available'


def build_family_round_record(
    *,
    round_index: int,
    family_source: str,
    validator_family_recipe: Mapping[str, Any] | None,
    validator_family_spec: Mapping[str, Any],
    warmup_summary: Mapping[str, Any],
    family_utility_summary: Mapping[str, Any],
    best_trial: Mapping[str, Any] | None,
    proxy_quality_summary: Mapping[str, Any],
    benchmark_summary: Mapping[str, Any],
    warmup_budget: Mapping[str, Any],
    family_aware_budget: Mapping[str, Any],
    invalid_count: int,
    execution_failure_count: int,
) -> Dict[str, Any]:
    record = {
        'round_index': round_index,
        'family_source': family_source,
        'family_hash': canonical_family_hash(validator_family_spec),
        'validator_family_recipe': dict(validator_family_recipe or {}),
        'validator_family_spec': dict(validator_family_spec),
        'u_family_online': family_utility_summary.get('u_family_online'),
        'stable_mean_advantage': family_utility_summary.get('stable_mean_advantage'),
        'stable_variance_advantage': family_utility_summary.get('stable_variance_advantage'),
        'family_split_mean_sep': family_utility_summary.get('family_split_mean_sep'),
        'family_split_worst_sep': family_utility_summary.get('family_split_worst_sep'),
        'best_trial': dict(best_trial or {}),
        'best_j_trial': (best_trial or {}).get('selection_score'),
        'proxy_alignment': proxy_quality_summary.get('proxy_alignment'),
        'j_system': benchmark_summary.get('j_system'),
        'warmup_summary': dict(warmup_summary),
        'warmup_paired_probe_budget': dict(warmup_budget),
        'family_aware_train_search_budget': dict(family_aware_budget),
        'proposal_invalid_count': invalid_count,
        'execution_failure_count': execution_failure_count,
    }
    record['strengths_weaknesses_summary'] = summarize_family_strengths(record)
    return record


def _sort_desc(records: Sequence[Mapping[str, Any]], key: str) -> List[Dict[str, Any]]:
    return sorted((dict(record) for record in records), key=lambda item: float(item.get(key) or float('-inf')), reverse=True)


def _redundant_families(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    redundant: List[Dict[str, Any]] = []
    for record in records:
        family_hash = str(record.get('family_hash') or '')
        if not family_hash:
            continue
        if family_hash in seen:
            redundant.append(
                {
                    'family_hash': family_hash,
                    'round_index': record.get('round_index'),
                    'matches_round_index': seen[family_hash].get('round_index'),
                    'proxy_alignment': record.get('proxy_alignment'),
                    'u_family_online': record.get('u_family_online'),
                }
            )
        else:
            seen[family_hash] = dict(record)
    return redundant


def _mismatch_families(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    for record in records:
        utility = _safe_float(record.get('u_family_online'))
        proxy = _safe_float(record.get('proxy_alignment'))
        if utility is None or proxy is None:
            continue
        if utility >= 0.0 and proxy < 0.3:
            mismatches.append(
                {
                    'round_index': record.get('round_index'),
                    'family_hash': record.get('family_hash'),
                    'u_family_online': utility,
                    'proxy_alignment': proxy,
                    'kind': 'high_online_low_offline',
                }
            )
        elif utility < 0.0 and proxy >= 0.5:
            mismatches.append(
                {
                    'round_index': record.get('round_index'),
                    'family_hash': record.get('family_hash'),
                    'u_family_online': utility,
                    'proxy_alignment': proxy,
                    'kind': 'low_online_interesting_offline',
                }
            )
    return mismatches


def build_family_memory_view(
    family_records: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    failure_k: int,
    current_round_index: int,
) -> Dict[str, Any]:
    prior_records = [dict(record) for record in family_records if int(record.get('round_index', 0)) < current_round_index]
    failed = [record for record in prior_records if int(record.get('execution_failure_count') or 0) > 0 or record.get('best_trial') in ({}, None)]
    memory_view = {
        'current_round_index': current_round_index,
        'num_prior_families': len(prior_records),
        'top_families_by_proxy_alignment': _sort_desc(prior_records, 'proxy_alignment')[:top_k],
        'top_families_by_j_system': _sort_desc(prior_records, 'j_system')[:top_k],
        'failed_families': _sort_desc(failed, 'execution_failure_count')[:failure_k],
        'redundant_families': _redundant_families(prior_records)[:failure_k],
        'online_offline_mismatches': _mismatch_families(prior_records)[: max(top_k, failure_k)],
    }
    return memory_view


def build_family_summary_table(family_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for record in family_records:
        rows.append(
            {
                'round_index': record.get('round_index'),
                'family_hash': record.get('family_hash'),
                'family_source': record.get('family_source'),
                'u_family_online': record.get('u_family_online'),
                'best_j_trial': record.get('best_j_trial'),
                'proxy_alignment': record.get('proxy_alignment'),
                'j_system': record.get('j_system'),
                'proposal_invalid_count': record.get('proposal_invalid_count'),
                'execution_failure_count': record.get('execution_failure_count'),
                'strengths_weaknesses_summary': record.get('strengths_weaknesses_summary'),
            }
        )
    return {'rows': rows}


def build_family_leaderboard(family_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(
        (dict(record) for record in family_records),
        key=lambda item: (
            float(item.get('proxy_alignment') if item.get('proxy_alignment') is not None else float('-inf')),
            float(item.get('j_system') if item.get('j_system') is not None else float('-inf')),
            float(item.get('u_family_online') if item.get('u_family_online') is not None else float('-inf')),
        ),
        reverse=True,
    )
    rows = []
    for rank, record in enumerate(ordered, start=1):
        rows.append(
            {
                'rank': rank,
                'round_index': record.get('round_index'),
                'family_hash': record.get('family_hash'),
                'family_source': record.get('family_source'),
                'proxy_alignment': record.get('proxy_alignment'),
                'u_family_online': record.get('u_family_online'),
                'j_system': record.get('j_system'),
                'best_j_trial': record.get('best_j_trial'),
                'summary': record.get('strengths_weaknesses_summary'),
            }
        )
    return {'rank_by': 'proxy_alignment', 'rows': rows}


def build_family_experience_notes(family_records: Sequence[Mapping[str, Any]]) -> str:
    lines = ['# Family Experience Notes', '']
    if not family_records:
        lines.append('- No family rounds completed yet.')
        return '\n'.join(lines) + '\n'
    leaderboard = build_family_leaderboard(family_records)['rows']
    lines.append('## Leaderboard')
    for row in leaderboard[:5]:
        lines.append(
            f"- round={row['round_index']} hash={row['family_hash']} ProxyAlignment={row['proxy_alignment']} U_family_online={row['u_family_online']} J_system={row['j_system']}: {row['summary']}"
        )
    mismatches = _mismatch_families(family_records)
    lines.append('')
    lines.append('## Mismatches')
    if mismatches:
        for item in mismatches[:10]:
            lines.append(
                f"- round={item['round_index']} hash={item['family_hash']} kind={item['kind']} U_family_online={item['u_family_online']} ProxyAlignment={item['proxy_alignment']}"
            )
    else:
        lines.append('- No strong online/offline mismatches recorded.')
    return '\n'.join(lines) + '\n'


def write_family_memory_artifacts(
    *,
    workdir: Path,
    family_records: Sequence[Mapping[str, Any]],
    current_memory_view: Mapping[str, Any] | None = None,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    family_history_path = workdir / 'family_history.jsonl'
    evolution_log_path = workdir / 'family_evolution_log.jsonl'
    with open(family_history_path, 'w', encoding='utf-8') as handle, open(evolution_log_path, 'w', encoding='utf-8') as evolution_handle:
        for record in family_records:
            line = json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + '\n'
            handle.write(line)
            evolution_handle.write(line)
    (workdir / 'family_summary_table.json').write_text(
        json.dumps(build_family_summary_table(family_records), indent=2, ensure_ascii=False, sort_keys=True),
        encoding='utf-8',
    )
    (workdir / 'family_leaderboard.json').write_text(
        json.dumps(build_family_leaderboard(family_records), indent=2, ensure_ascii=False, sort_keys=True),
        encoding='utf-8',
    )
    (workdir / 'family_experience_notes.md').write_text(build_family_experience_notes(family_records), encoding='utf-8')
    if current_memory_view is not None:
        (workdir / 'family_memory_view.json').write_text(
            json.dumps(dict(current_memory_view), indent=2, ensure_ascii=False, sort_keys=True),
            encoding='utf-8',
        )
