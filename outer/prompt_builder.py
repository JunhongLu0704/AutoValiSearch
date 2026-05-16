from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from training.validation_protocols import FAMILY_AGGREGATION, FAMILY_ALPHA, get_stage1_group_library


def _family_summary(validator_family_spec: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not validator_family_spec:
        return None
    return {
        'include_vs': bool(validator_family_spec.get('include_vs', True)),
        'family_aggregation': validator_family_spec.get('family_aggregation', FAMILY_AGGREGATION),
        'alpha': validator_family_spec.get('alpha', FAMILY_ALPHA),
        'members': [
            {
                'name': member.get('name'),
                'protocol': member.get('protocol'),
                'groups': [group.get('name') for group in (member.get('spec') or {}).get('groups', [])],
            }
            for member in validator_family_spec.get('validators', [])
        ],
    }


def build_validator_prompt_bundle(
    *,
    validator_role_id: str,
    fixed_config: Mapping[str, Any],
    warmup_summary: Mapping[str, Any] | None,
    family_memory_view: Mapping[str, Any] | None,
    round_index: int,
) -> Dict[str, str]:
    group_library = get_stage1_group_library()
    system_prompt = (
        'You are Val Agent for AutoValiSearch Stage 2. '
        'Return only one JSON object. Do not output markdown, prose, or code fences.'
    )
    user_payload = {
        'task': 'Design one validator_family_recipe for the current round.',
        'project_name': 'AutoValiSearch',
        'stage': 'Stage 2',
        'round_index': round_index,
        'objective': (
            'Choose up to 2 validator-family members that are likely to yield a better search-time proxy for '
            'unseen-domain generalization under the bounded validator DSL, informed by prior family-memory summaries.'
        ),
        'required_output_schema': {
            'validator_role_id': validator_role_id,
            'hypothesis': 'short string',
            'validator_protocol': 'validator_family',
            'validator_family_recipe': {
                'include_vs': True,
                'members': [
                    {
                        'name': 'member_name',
                        'protocol': 'handcrafted_va or llm_va',
                        'groups': [
                            {'name': 'group_name', 'ops': ['must match one allowed group template exactly']}
                        ],
                    }
                ],
            },
        },
        'rules': {
            'max_members': 2,
            'allowed_member_protocols': ['handcrafted_va', 'llm_va'],
            'family_aggregation': FAMILY_AGGREGATION,
            'alpha': FAMILY_ALPHA,
            'stage1_group_templates': list(group_library.values()),
            'forbidden': [
                'TCV',
                'arbitrary code',
                'extra top-level keys',
                'more than 2 members',
                'group ops outside the allowed templates',
                'raw history file references',
            ],
        },
        'fixed_run_context': {
            'dataset': fixed_config.get('dataset'),
            'budget': fixed_config.get('budget'),
            'eval_split_dirs': fixed_config.get('eval_split_dirs'),
            'eval_seeds': fixed_config.get('eval_seeds'),
            'warmup_probe_count': fixed_config.get('warmup_probe_count'),
        },
        'warmup_summary': warmup_summary,
        'family_memory_view': family_memory_view,
    }
    return {
        'system_prompt': system_prompt,
        'user_prompt': json.dumps(user_payload, ensure_ascii=False, indent=2, sort_keys=True),
    }


def build_validator_repair_prompt_bundle(
    *,
    validator_role_id: str,
    previous_output: Mapping[str, Any] | str,
    validation_error_report: Mapping[str, Any],
    family_memory_view: Mapping[str, Any] | None,
    round_index: int,
) -> Dict[str, str]:
    system_prompt = (
        'You are Val Agent for AutoValiSearch Stage 2. Return corrected JSON only. '
        'Keep the same intent and fix only invalid structure.'
    )
    user_payload = {
        'task': 'Repair the invalid validator_family_recipe output.',
        'validator_role_id': validator_role_id,
        'round_index': round_index,
        'family_memory_view': family_memory_view,
        'previous_output': previous_output,
        'validation_error_report': validation_error_report,
    }
    return {
        'system_prompt': system_prompt,
        'user_prompt': json.dumps(user_payload, ensure_ascii=False, indent=2, sort_keys=True),
    }


def build_train_prompt_bundle(
    *,
    proposal_id: str,
    fixed_config: Mapping[str, Any],
    tunables: Mapping[str, Sequence[Any]],
    locked_family_spec: Mapping[str, Any],
    family_utility_summary: Mapping[str, Any] | None,
    best_trials: Sequence[Mapping[str, Any]],
    recent_failures: Sequence[Mapping[str, Any]],
    round_index: int,
) -> Dict[str, str]:
    system_prompt = (
        'You are Train Agent for AutoValiSearch Stage 2. Return only one JSON object with proposal_id, hypothesis, and config.'
    )
    user_payload = {
        'task': 'Propose one training configuration under the locked validator family.',
        'project_name': 'AutoValiSearch',
        'stage': 'Stage 2',
        'round_index': round_index,
        'optimization_target': 'J_trial',
        'optimization_target_definition': 'J_trial = harmonic_mean(split_mean, split_worst) over child eval selection scores.',
        'proposal_schema': {
            'proposal_id': proposal_id,
            'hypothesis': 'short string',
            'config': {key: list(values) for key, values in tunables.items()},
        },
        'fixed_run_context': {
            'dataset': fixed_config.get('dataset'),
            'budget': fixed_config.get('budget'),
            'bn_mode': fixed_config.get('bn_mode'),
            'amp': fixed_config.get('amp'),
            'disturb_mode': fixed_config.get('disturb_mode'),
            'eval_split_dirs': fixed_config.get('eval_split_dirs'),
            'eval_seeds': fixed_config.get('eval_seeds'),
        },
        'locked_validator_family': _family_summary(locked_family_spec),
        'family_utility_summary': family_utility_summary,
        'best_trials': list(best_trials),
        'recent_failures': list(recent_failures),
        'rules': {
            'optimize': 'J_trial only',
            'do_not_optimize': ['best_val_acc1', 'ProxyAlignment', 'J_system', 'test metrics'],
            'higher_is_better': True,
        },
    }
    return {
        'system_prompt': system_prompt,
        'user_prompt': json.dumps(user_payload, ensure_ascii=False, indent=2, sort_keys=True),
    }

