from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

from training.validation_protocols import (
    FAMILY_AGGREGATION,
    FAMILY_ALPHA,
    MAX_VALIDATOR_FAMILY_MEMBERS,
    VALID_FAMILY_MEMBER_PROTOCOLS,
    VALIDATOR_FAMILY_PRESET,
    ValidatorSpecError,
    build_builtin_validator_family_spec,
    get_stage1_group_library,
    normalize_validator_family_spec,
)


class ValidatorRoleError(ValueError):
    pass


def _canonical_group_signature(ops: list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(op).strip().lower() for op in ops))


def _group_registry_by_signature() -> Dict[tuple[str, ...], Dict[str, Any]]:
    registry = {}
    for spec in get_stage1_group_library().values():
        registry[_canonical_group_signature(list(spec['ops']))] = spec
    return registry


def build_builtin_validator_role_output(
    *,
    validator_role_id: str,
    validator_protocol: str,
    validator_preset: Optional[str],
) -> Dict[str, Any]:
    if str(validator_protocol).lower() != 'validator_family':
        raise ValidatorRoleError('Built-in validator role output only supports validator_family in Stage 1')
    spec = build_builtin_validator_family_spec()
    recipe_members = []
    for member in spec['validators']:
        recipe_members.append(
            {
                'name': member['name'],
                'protocol': member['protocol'],
                'groups': [
                    {
                        'name': group['name'],
                        'ops': list(group['ops']),
                    }
                    for group in member['spec']['groups']
                ],
            }
        )
    return {
        'validator_role_id': validator_role_id,
        'hypothesis': 'Use a deterministic validator family with one photometric member and one structure/noise member.',
        'validator_protocol': 'validator_family',
        'validator_family_recipe': {
            'include_vs': bool(spec.get('include_vs', True)),
            'members': recipe_members,
        },
    }


def compile_validator_family_recipe(
    recipe: Mapping[str, Any],
    *,
    validator_preset: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(recipe, Mapping):
        raise ValidatorRoleError('validator_family_recipe must be an object')
    include_vs = bool(recipe.get('include_vs', True))
    members_raw = list(recipe.get('members') or [])
    if not members_raw:
        raise ValidatorRoleError('validator_family_recipe.members must contain at least one member')
    if len(members_raw) > MAX_VALIDATOR_FAMILY_MEMBERS:
        raise ValidatorRoleError(
            f'validator_family_recipe.members exceeds max size {MAX_VALIDATOR_FAMILY_MEMBERS}'
        )
    group_registry = _group_registry_by_signature()
    normalized_members = []
    seen_names = set()
    for index, member in enumerate(members_raw):
        if not isinstance(member, Mapping):
            raise ValidatorRoleError(f'member[{index}] must be an object')
        member_name = str(member.get('name', '')).strip()
        if not member_name:
            raise ValidatorRoleError(f'member[{index}].name must be non-empty')
        if member_name in seen_names:
            raise ValidatorRoleError(f'duplicate family member name: {member_name}')
        seen_names.add(member_name)
        protocol = str(member.get('protocol', '')).strip().lower()
        if protocol not in VALID_FAMILY_MEMBER_PROTOCOLS:
            raise ValidatorRoleError(
                f'member[{index}].protocol must be one of {sorted(VALID_FAMILY_MEMBER_PROTOCOLS)}'
            )
        groups_raw = list(member.get('groups') or [])
        if not groups_raw:
            raise ValidatorRoleError(f'member[{index}].groups must contain at least one group')
        normalized_groups = []
        seen_group_names = set()
        for group_index, group in enumerate(groups_raw):
            if not isinstance(group, Mapping):
                raise ValidatorRoleError(f'member[{index}].groups[{group_index}] must be an object')
            group_name = str(group.get('name', '')).strip()
            if not group_name:
                raise ValidatorRoleError(f'member[{index}].groups[{group_index}].name must be non-empty')
            if group_name in seen_group_names:
                raise ValidatorRoleError(f'duplicate group name within member {member_name}: {group_name}')
            seen_group_names.add(group_name)
            ops = [str(op).strip().lower() for op in list(group.get('ops') or [])]
            if not ops:
                raise ValidatorRoleError(f'member[{index}].groups[{group_index}].ops must contain at least one op')
            signature = _canonical_group_signature(ops)
            if signature not in group_registry:
                allowed = [list(spec['ops']) for spec in get_stage1_group_library().values()]
                raise ValidatorRoleError(
                    f'member[{index}].groups[{group_index}] ops must match one Stage 1 group template; allowed={allowed}'
                )
            template = copy.deepcopy(group_registry[signature])
            normalized_groups.append(template)
        normalized_members.append(
            {
                'name': member_name,
                'protocol': protocol,
                'spec': {
                    'preset': 'autovalisearch_compiled_v1',
                    'groups': normalized_groups,
                    'per_image_policy': 'one_per_group',
                    'aggregation': 'harmonic_avg_worst',
                },
            }
        )
    compiled = {
        'preset': str(validator_preset or VALIDATOR_FAMILY_PRESET),
        'include_vs': include_vs,
        'family_aggregation': FAMILY_AGGREGATION,
        'alpha': FAMILY_ALPHA,
        'validators': normalized_members,
    }
    try:
        return normalize_validator_family_spec(raw_spec=compiled, validator_preset=validator_preset)
    except ValidatorSpecError as exc:
        raise ValidatorRoleError(str(exc)) from exc


def normalize_validator_role_output(
    payload: Mapping[str, Any],
    *,
    validator_protocol: str,
    validator_preset: Optional[str],
) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidatorRoleError('validator role output must be a JSON object')
    protocol = str(payload.get('validator_protocol') or validator_protocol).lower()
    if protocol != 'validator_family' or str(validator_protocol).lower() != 'validator_family':
        raise ValidatorRoleError('Stage 1 validator role only supports validator_family')
    role_id = str(payload.get('validator_role_id', '')).strip()
    if not role_id:
        raise ValidatorRoleError('validator_role_id must be non-empty')
    hypothesis = str(payload.get('hypothesis', '')).strip()
    if not hypothesis:
        raise ValidatorRoleError('hypothesis must be non-empty')
    recipe = payload.get('validator_family_recipe')
    if recipe is None:
        raise ValidatorRoleError('validator_family_recipe is required')
    compiled = compile_validator_family_recipe(recipe, validator_preset=validator_preset)
    return {
        'validator_role_id': role_id,
        'hypothesis': hypothesis,
        'validator_protocol': 'validator_family',
        'validator_family_recipe': copy.deepcopy(recipe),
        'validator_family_spec': compiled,
    }


def build_validation_error_report(exc: Exception) -> Dict[str, Any]:
    group_templates = [list(spec['ops']) for spec in get_stage1_group_library().values()]
    return {
        'status': 'invalid',
        'error_type': exc.__class__.__name__,
        'message': str(exc),
        'allowed_rules_summary': {
            'protocol': 'validator_family',
            'max_members': MAX_VALIDATOR_FAMILY_MEMBERS,
            'allowed_member_protocols': sorted(VALID_FAMILY_MEMBER_PROTOCOLS),
            'include_vs': 'boolean',
            'allowed_group_templates': group_templates,
            'family_aggregation': FAMILY_AGGREGATION,
            'alpha': FAMILY_ALPHA,
        },
        'repair_instruction': 'Return corrected JSON only. Keep the same intent. Fix only invalid fields.',
    }

