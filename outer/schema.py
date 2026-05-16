from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from training.inner_loop import compute_config_hash, normalize_config, validate_config
from training.validation_protocols import VALIDATOR_FAMILY_PRESET

ROOT = Path(__file__).resolve().parents[1]
HASH_PLACEHOLDER_DIR = ROOT / '_outer_hash_placeholder'
REQUIRED_FIXED_FIELDS = ('dataset', 'split_dir', 'image_root', 'budget', 'seed')
DEFAULT_TUNABLES = {
    'lr': (0.001, 0.003, 0.005),
    'lambdap': (1.0, 2.0, 4.0, 8.0),
    'epochp': (1, 2, 4),
    'weight_decay': (0.0, 1e-4, 5e-4),
}
BASE_REQUIRED_FIELDS = ('lr', 'lambdap', 'epochp', 'weight_decay')
TOP_LEVEL_KEYS = {'proposal_id', 'hypothesis', 'config'}


class ProposalError(ValueError):
    pass


@dataclass
class SearchSpace:
    fixed_config: Dict[str, Any]
    tunables: Dict[str, Tuple[Any, ...]]

    def allowed_config_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.tunables.keys()) | set(self.fixed_config.keys())))


def _coerce_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if lowered in {'0', 'false', 'no', 'n', 'off'}:
            return False
    raise ProposalError(f'Invalid boolean value for {field}: {value!r}')


def _normalize_fixed_config(raw_fixed_config: Mapping[str, Any]) -> Dict[str, Any]:
    fixed = dict(raw_fixed_config)
    missing = [field for field in REQUIRED_FIXED_FIELDS if field not in fixed]
    if missing:
        raise ProposalError(f'Missing fixed config fields: {missing}')
    fixed['dataset'] = str(fixed['dataset']).upper()
    fixed['split_dir'] = os.path.abspath(str(fixed['split_dir']))
    fixed['image_root'] = os.path.abspath(str(fixed['image_root']))
    fixed['budget'] = str(fixed['budget']).lower()
    fixed['seed'] = int(fixed['seed'])
    fixed['bn_mode'] = str(fixed.get('bn_mode', 'eval')).lower()
    fixed['amp'] = _coerce_bool(fixed.get('amp', True), field='amp')
    fixed['workers'] = int(fixed.get('workers', 4))
    fixed['prefetch_factor'] = int(fixed.get('prefetch_factor', 2))
    fixed['bs'] = int(fixed.get('bs', 128))
    fixed['disturb_mode'] = str(fixed.get('disturb_mode', 'rsw')).lower()
    fixed['validator_protocol'] = str(fixed.get('validator_protocol', 'validator_family')).lower()
    fixed['validator_preset'] = str(fixed.get('validator_preset') or 'llm_family_v1')
    fixed['rsw_min'] = float(fixed.get('rsw_min', 0.2))
    fixed['aggregate_objective'] = 'j_trial'
    eval_split_dirs = fixed.get('eval_split_dirs') or [fixed['split_dir']]
    fixed['eval_split_dirs'] = [os.path.abspath(str(item)) for item in eval_split_dirs]
    eval_seeds = fixed.get('eval_seeds') or [fixed['seed']]
    fixed['eval_seeds'] = [int(item) for item in eval_seeds]
    fixed['per_split_parallel_evals'] = max(int(fixed.get('per_split_parallel_evals', 2)), 1)
    fixed['validator_family_spec'] = copy.deepcopy(fixed.get('validator_family_spec'))
    fixed['validator_spec'] = copy.deepcopy(fixed.get('validator_spec'))
    fixed['warmup_probe_count'] = int(fixed.get('warmup_probe_count', 3))
    fixed['u_family_gamma'] = float(fixed.get('u_family_gamma', 0.5))
    fixed['design_retry_limit'] = int(fixed.get('design_retry_limit', 3))
    fixed['max_family_rounds'] = max(int(fixed.get('max_family_rounds', 1)), 1)
    fixed['family_memory_top_k'] = max(int(fixed.get('family_memory_top_k', 3)), 1)
    fixed['family_memory_failure_k'] = max(int(fixed.get('family_memory_failure_k', 3)), 1)
    fixed['stage1_project_name'] = 'AutoValiSearch'
    fixed['stage2_project_name'] = 'AutoValiSearch'
    return fixed


def default_search_space(fixed_config: Mapping[str, Any]) -> SearchSpace:
    return SearchSpace(fixed_config=_normalize_fixed_config(fixed_config), tunables=copy.deepcopy(DEFAULT_TUNABLES))


def validate_proposal_payload(proposal: Mapping[str, Any]) -> None:
    if not isinstance(proposal, Mapping):
        raise ProposalError('Proposal must be a JSON object')
    missing = TOP_LEVEL_KEYS - set(proposal.keys())
    extra = set(proposal.keys()) - TOP_LEVEL_KEYS
    if missing:
        raise ProposalError(f'Proposal is missing top-level keys: {sorted(missing)}')
    if extra:
        raise ProposalError(f'Proposal contains unsupported top-level keys: {sorted(extra)}')
    if not str(proposal['proposal_id']).strip():
        raise ProposalError('proposal_id must be a non-empty string')
    if not str(proposal['hypothesis']).strip():
        raise ProposalError('hypothesis must be a non-empty string')
    if not isinstance(proposal['config'], Mapping):
        raise ProposalError('config must be a JSON object')


def _normalize_echo_value(field: str, value: Any) -> Any:
    if field == 'dataset':
        return str(value).upper()
    if field in {'split_dir', 'image_root'}:
        return os.path.abspath(str(value))
    if field == 'budget':
        return str(value).lower()
    if field == 'seed':
        return int(value)
    if field == 'bn_mode':
        return str(value).lower()
    if field == 'amp':
        return _coerce_bool(value, field=field)
    if field in {'workers', 'prefetch_factor', 'bs', 'warmup_probe_count', 'design_retry_limit', 'max_family_rounds', 'family_memory_top_k', 'family_memory_failure_k', 'per_split_parallel_evals'}:
        return int(value)
    if field in {'rsw_min', 'u_family_gamma'}:
        return float(value)
    if field == 'disturb_mode':
        return str(value).lower()
    if field == 'validator_protocol':
        return str(value).lower()
    if field == 'validator_preset':
        return str(value)
    if field == 'eval_split_dirs':
        return [os.path.abspath(str(item)) for item in value]
    if field == 'eval_seeds':
        return [int(item) for item in value]
    return value


def _build_trial_request_config(proposal_config: Mapping[str, Any], search_space: SearchSpace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(proposal_config, Mapping):
        raise ProposalError('Proposal config must be a JSON object')
    config = dict(proposal_config)
    extra = set(config.keys()) - set(search_space.allowed_config_keys())
    if extra:
        raise ProposalError(f'Proposal config contains unsupported keys: {sorted(extra)}')
    notes: Dict[str, Any] = {'echoed_fixed_fields': []}
    merged = dict(search_space.fixed_config)
    for field, fixed_value in search_space.fixed_config.items():
        if field in config:
            normalized = _normalize_echo_value(field, config[field])
            if normalized != fixed_value:
                raise ProposalError(f'Proposal attempted to override fixed field {field}: {normalized!r} != {fixed_value!r}')
            notes['echoed_fixed_fields'].append(field)
            config.pop(field)
    missing = [field for field in BASE_REQUIRED_FIELDS if field not in config]
    if missing:
        raise ProposalError(f'Proposal config is missing required fields: {missing}')
    config['lr'] = float(config['lr'])
    config['lambdap'] = float(config['lambdap'])
    config['epochp'] = int(config['epochp'])
    config['weight_decay'] = float(config['weight_decay'])
    for field in BASE_REQUIRED_FIELDS:
        allowed = search_space.tunables[field]
        if config[field] not in allowed:
            raise ProposalError(f'Invalid value for {field}: {config[field]!r}; allowed={list(allowed)!r}')
    merged.update(config)
    if not notes['echoed_fixed_fields']:
        notes.pop('echoed_fixed_fields')
    return merged, notes


def resolve_proposal(proposal: Mapping[str, Any], search_space: SearchSpace, trial_dir: str) -> Dict[str, Any]:
    validate_proposal_payload(proposal)
    merged_raw_config, normalization_notes = _build_trial_request_config(proposal['config'], search_space)
    normalized_config = normalize_config(merged_raw_config, trial_dir)
    validate_config(normalized_config)
    config_hash = compute_config_hash(normalized_config)
    normalized_config['config_hash'] = config_hash
    return {
        'proposal_id': str(proposal['proposal_id']),
        'hypothesis': str(proposal['hypothesis']).strip(),
        'proposal_config': dict(proposal['config']),
        'raw_config': merged_raw_config,
        'normalized_config': normalized_config,
        'normalization_notes': normalization_notes,
        'config_hash': config_hash,
    }


def compute_candidate_hash(proposal_config: Mapping[str, Any], search_space: SearchSpace) -> str:
    merged_raw_config, _ = _build_trial_request_config(proposal_config, search_space)
    normalized_config = normalize_config(merged_raw_config, str(HASH_PLACEHOLDER_DIR))
    validate_config(normalized_config)
    return compute_config_hash(normalized_config)


def clone_with_locked_family(search_space: SearchSpace, validator_family_spec: Mapping[str, Any]) -> SearchSpace:
    fixed = dict(search_space.fixed_config)
    fixed['validator_protocol'] = 'validator_family'
    fixed['validator_preset'] = str(fixed.get('validator_preset') or VALIDATOR_FAMILY_PRESET)
    fixed['validator_family_spec'] = copy.deepcopy(dict(validator_family_spec))
    fixed['validator_spec'] = None
    return SearchSpace(fixed_config=fixed, tunables=copy.deepcopy(search_space.tunables))

