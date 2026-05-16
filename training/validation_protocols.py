from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from dataset.datasets import CustomDataset

VALID_VALIDATOR_PROTOCOLS = {'vs', 'handcrafted_va', 'llm_va', 'validator_family'}
HANDCRAFTED_VA_PRESET = 'handcrafted_v1'
LLM_VA_PRESET = 'llm_v1'
WARMUP_PROBE_PRESET = 'warmup_probe_v1'
VALIDATOR_FAMILY_PRESET = 'handcrafted_family_v1'
VALID_FAMILY_AGGREGATIONS = {'vs_plus_harmonic_mean_min'}
VALID_VA_AGGREGATIONS = {'harmonic_avg_worst'}
VALID_PER_IMAGE_POLICIES = {'one_per_group'}
VALID_LLM_VA_OPS = {
    'autocontrast',
    'brightness',
    'contrast',
    'gaussian_blur',
    'gaussian_noise',
    'grayscale',
    'posterize',
    'saturation',
    'sharpness',
}
VALID_FAMILY_MEMBER_PROTOCOLS = {'handcrafted_va', 'llm_va'}
MAX_VALIDATOR_GROUPS = 4
MAX_GROUP_OPS = 3
MAX_SEVERITY = 3
MAX_VALIDATOR_FAMILY_MEMBERS = 2
FAMILY_ALPHA = 0.1
FAMILY_AGGREGATION = 'vs_plus_harmonic_mean_min'

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

STAGE1_GROUP_LIBRARY: Dict[str, Dict[str, Any]] = {
    'photometric_family': {
        'name': 'photometric_family',
        'ops': ['brightness', 'contrast', 'saturation'],
        'sample_k': 1,
        'severity_min': 1,
        'severity_max': 2,
    },
    'blur_noise_family': {
        'name': 'blur_noise_family',
        'ops': ['gaussian_blur', 'gaussian_noise'],
        'sample_k': 1,
        'severity_min': 1,
        'severity_max': 2,
    },
    'structure_family': {
        'name': 'structure_family',
        'ops': ['grayscale', 'posterize'],
        'sample_k': 1,
        'severity_min': 1,
        'severity_max': 2,
    },
    'tone_family': {
        'name': 'tone_family',
        'ops': ['autocontrast', 'sharpness'],
        'sample_k': 1,
        'severity_min': 1,
        'severity_max': 2,
    },
}


class ValidatorSpecError(ValueError):
    pass


class GaussianNoiseTensor:
    def __init__(self, std: float):
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp(0.0, 1.0)


def build_standard_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_stage1_group_library() -> Dict[str, Dict[str, Any]]:
    return {name: dict(spec) for name, spec in STAGE1_GROUP_LIBRARY.items()}


def build_handcrafted_validator_spec(preset: str = HANDCRAFTED_VA_PRESET) -> Dict[str, Any]:
    preset = str(preset)
    if preset in {HANDCRAFTED_VA_PRESET, WARMUP_PROBE_PRESET}:
        groups = [
            dict(STAGE1_GROUP_LIBRARY['photometric_family']),
            dict(STAGE1_GROUP_LIBRARY['blur_noise_family']),
            dict(STAGE1_GROUP_LIBRARY['structure_family']),
            dict(STAGE1_GROUP_LIBRARY['tone_family']),
        ]
    elif preset == 'handcrafted_family_member_a_v1':
        groups = [
            dict(STAGE1_GROUP_LIBRARY['photometric_family']),
            dict(STAGE1_GROUP_LIBRARY['tone_family']),
        ]
    elif preset == 'handcrafted_family_member_b_v1':
        groups = [
            dict(STAGE1_GROUP_LIBRARY['blur_noise_family']),
            dict(STAGE1_GROUP_LIBRARY['structure_family']),
        ]
    else:
        raise ValidatorSpecError(f'Unsupported handcrafted validator preset: {preset}')
    return {
        'preset': preset,
        'groups': groups,
        'per_image_policy': 'one_per_group',
        'aggregation': 'harmonic_avg_worst',
    }


def build_default_llm_validator_spec() -> Dict[str, Any]:
    return {
        'preset': LLM_VA_PRESET,
        'groups': [
            dict(STAGE1_GROUP_LIBRARY['photometric_family']),
            dict(STAGE1_GROUP_LIBRARY['blur_noise_family']),
            dict(STAGE1_GROUP_LIBRARY['structure_family']),
        ],
        'per_image_policy': 'one_per_group',
        'aggregation': 'harmonic_avg_worst',
    }


def build_warmup_probe_validator_spec() -> Dict[str, Any]:
    return build_handcrafted_validator_spec(WARMUP_PROBE_PRESET)


def build_builtin_validator_family_spec() -> Dict[str, Any]:
    return {
        'preset': VALIDATOR_FAMILY_PRESET,
        'include_vs': True,
        'family_aggregation': FAMILY_AGGREGATION,
        'alpha': FAMILY_ALPHA,
        'validators': [
            {
                'name': 'photometric_robustness',
                'protocol': 'handcrafted_va',
                'spec': build_handcrafted_validator_spec('handcrafted_family_member_a_v1'),
            },
            {
                'name': 'structure_robustness',
                'protocol': 'handcrafted_va',
                'spec': build_handcrafted_validator_spec('handcrafted_family_member_b_v1'),
            },
        ],
    }


def _validate_group(group: Mapping[str, Any], index: int) -> Dict[str, Any]:
    name = str(group.get('name', '')).strip()
    if not name:
        raise ValidatorSpecError(f'Validator group at index {index} is missing a non-empty name')
    ops = [str(op).strip() for op in list(group.get('ops') or [])]
    if not ops:
        raise ValidatorSpecError(f'Validator group {name} must declare at least one op')
    if len(ops) > MAX_GROUP_OPS:
        raise ValidatorSpecError(f'Validator group {name} exceeds max ops per group: {MAX_GROUP_OPS}')
    invalid_ops = [op for op in ops if op not in VALID_LLM_VA_OPS]
    if invalid_ops:
        raise ValidatorSpecError(f'Validator group {name} contains unsupported ops: {invalid_ops}')
    sample_k = int(group.get('sample_k', 1))
    if sample_k <= 0 or sample_k > len(ops):
        raise ValidatorSpecError(f'Validator group {name} has invalid sample_k={sample_k}')
    severity_min = int(group.get('severity_min', 1))
    severity_max = int(group.get('severity_max', severity_min))
    if not (1 <= severity_min <= MAX_SEVERITY and 1 <= severity_max <= MAX_SEVERITY):
        raise ValidatorSpecError(f'Validator group {name} severity must lie in [1, {MAX_SEVERITY}]')
    if severity_min > severity_max:
        raise ValidatorSpecError(f'Validator group {name} has severity_min > severity_max')
    return {
        'name': name,
        'ops': ops,
        'sample_k': sample_k,
        'severity_min': severity_min,
        'severity_max': severity_max,
    }


def normalize_validator_spec(
    validator_protocol: str,
    *,
    validator_preset: Optional[str] = None,
    raw_spec: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    protocol = str(validator_protocol).lower()
    if protocol not in VALID_VALIDATOR_PROTOCOLS:
        raise ValidatorSpecError(f'Unsupported validator_protocol: {validator_protocol}')
    if protocol == 'vs':
        return None
    if protocol == 'validator_family':
        raise ValidatorSpecError('validator_family requires normalize_validator_family_spec')
    if protocol == 'handcrafted_va':
        preset = str(validator_preset or HANDCRAFTED_VA_PRESET)
        spec_source = dict(raw_spec) if raw_spec is not None else build_handcrafted_validator_spec(preset)
    else:
        spec_source = dict(raw_spec) if raw_spec is not None else build_default_llm_validator_spec()
    groups_raw = list(spec_source.get('groups') or [])
    if not groups_raw:
        raise ValidatorSpecError(f'{protocol} validator_spec must define at least one group')
    if len(groups_raw) > MAX_VALIDATOR_GROUPS:
        raise ValidatorSpecError(f'{protocol} validator_spec exceeds max groups: {MAX_VALIDATOR_GROUPS}')
    groups = [_validate_group(group, index) for index, group in enumerate(groups_raw)]
    group_names = [group['name'] for group in groups]
    if len(group_names) != len(set(group_names)):
        raise ValidatorSpecError(f'{protocol} validator_spec group names must be unique')
    per_image_policy = str(spec_source.get('per_image_policy', 'one_per_group')).lower()
    if per_image_policy not in VALID_PER_IMAGE_POLICIES:
        raise ValidatorSpecError(f'Unsupported per_image_policy: {per_image_policy}')
    aggregation = str(spec_source.get('aggregation', 'harmonic_avg_worst')).lower()
    if aggregation not in VALID_VA_AGGREGATIONS:
        raise ValidatorSpecError(f'Unsupported aggregation: {aggregation}')
    return {
        'preset': str(spec_source.get('preset') or validator_preset or LLM_VA_PRESET),
        'groups': groups,
        'per_image_policy': per_image_policy,
        'aggregation': aggregation,
    }


def normalize_validator_family_spec(
    *,
    raw_spec: Optional[Mapping[str, Any]] = None,
    validator_preset: Optional[str] = None,
) -> Dict[str, Any]:
    spec_source = dict(raw_spec) if raw_spec is not None else build_builtin_validator_family_spec()
    include_vs = bool(spec_source.get('include_vs', True))
    family_aggregation = str(spec_source.get('family_aggregation', FAMILY_AGGREGATION)).lower()
    if family_aggregation != FAMILY_AGGREGATION:
        raise ValidatorSpecError(f'Unsupported family_aggregation: {family_aggregation}')
    alpha = float(spec_source.get('alpha', FAMILY_ALPHA))
    if abs(alpha - FAMILY_ALPHA) > 1e-8:
        raise ValidatorSpecError(f'validator_family alpha must equal {FAMILY_ALPHA}')
    validators_raw = list(spec_source.get('validators') or [])
    if not validators_raw:
        raise ValidatorSpecError('validator_family must define at least one augmentation validator')
    if len(validators_raw) > MAX_VALIDATOR_FAMILY_MEMBERS:
        raise ValidatorSpecError(
            f'validator_family exceeds max augmentation validators: {MAX_VALIDATOR_FAMILY_MEMBERS}'
        )
    validators = []
    names = set()
    for index, member in enumerate(validators_raw):
        if not isinstance(member, Mapping):
            raise ValidatorSpecError(f'validator_family member at index {index} must be an object')
        name = str(member.get('name', '')).strip()
        if not name:
            raise ValidatorSpecError(f'validator_family member at index {index} is missing a non-empty name')
        if name in names:
            raise ValidatorSpecError('validator_family member names must be unique')
        names.add(name)
        protocol = str(member.get('protocol', '')).strip().lower()
        if protocol not in VALID_FAMILY_MEMBER_PROTOCOLS:
            raise ValidatorSpecError(f'validator_family member {name} uses unsupported protocol: {protocol}')
        normalized_member_spec = normalize_validator_spec(
            protocol,
            validator_preset=member.get('preset'),
            raw_spec=member.get('spec'),
        )
        validators.append({'name': name, 'protocol': protocol, 'spec': normalized_member_spec})
    return {
        'preset': str(spec_source.get('preset') or validator_preset or VALIDATOR_FAMILY_PRESET),
        'include_vs': include_vs,
        'family_aggregation': FAMILY_AGGREGATION,
        'alpha': FAMILY_ALPHA,
        'validators': validators,
    }


def _severity_level(group: Mapping[str, Any]) -> int:
    severity_min = int(group.get('severity_min', 1))
    severity_max = int(group.get('severity_max', severity_min))
    return max(severity_min, min(MAX_SEVERITY, severity_max))


def _apply_pil_op(op: str, severity: int):
    if op == 'brightness':
        factor = {1: 1.10, 2: 1.20, 3: 1.30}[severity]
        return transforms.Lambda(lambda image: TF.adjust_brightness(image, factor))
    if op == 'contrast':
        factor = {1: 1.15, 2: 1.30, 3: 1.45}[severity]
        return transforms.Lambda(lambda image: TF.adjust_contrast(image, factor))
    if op == 'saturation':
        factor = {1: 1.15, 2: 1.30, 3: 1.45}[severity]
        return transforms.Lambda(lambda image: TF.adjust_saturation(image, factor))
    if op == 'gaussian_blur':
        sigma = {1: 0.8, 2: 1.2, 3: 1.6}[severity]
        return transforms.GaussianBlur(kernel_size=5, sigma=sigma)
    if op == 'grayscale':
        return transforms.Grayscale(num_output_channels=3)
    if op == 'posterize':
        bits = {1: 6, 2: 5, 3: 4}[severity]
        return transforms.Lambda(lambda image: TF.autocontrast(TF.posterize(image, bits)))
    if op == 'sharpness':
        factor = {1: 1.5, 2: 2.0, 3: 2.5}[severity]
        return transforms.Lambda(lambda image: TF.adjust_sharpness(image, factor))
    if op == 'autocontrast':
        return transforms.Lambda(TF.autocontrast)
    return None


def _apply_tensor_op(op: str, severity: int):
    if op == 'gaussian_noise':
        std = {1: 0.02, 2: 0.04, 3: 0.06}[severity]
        return GaussianNoiseTensor(std)
    return None


def _build_group_transform(group: Mapping[str, Any]) -> transforms.Compose:
    severity = _severity_level(group)
    ops = list(group['ops'])[: int(group.get('sample_k', 1))]
    pre_tensor_ops = []
    post_tensor_ops = []
    for op in ops:
        pil_op = _apply_pil_op(op, severity)
        if pil_op is not None:
            pre_tensor_ops.append(pil_op)
            continue
        tensor_op = _apply_tensor_op(op, severity)
        if tensor_op is not None:
            post_tensor_ops.append(tensor_op)
            continue
        raise ValidatorSpecError(f'Unsupported validator op during transform build: {op}')
    return transforms.Compose([
        transforms.Resize((224, 224)),
        *pre_tensor_ops,
        transforms.ToTensor(),
        *post_tensor_ops,
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def build_llm_va_group_transforms(validator_spec: Mapping[str, Any]) -> Dict[str, transforms.Compose]:
    normalized = normalize_validator_spec('llm_va', raw_spec=validator_spec)
    return {group['name']: _build_group_transform(group) for group in normalized['groups']}


def build_handcrafted_va_group_transforms(
    validator_spec: Optional[Mapping[str, Any]] = None,
    *,
    validator_preset: str = HANDCRAFTED_VA_PRESET,
) -> Dict[str, transforms.Compose]:
    spec = normalize_validator_spec('handcrafted_va', validator_preset=validator_preset, raw_spec=validator_spec)
    return {group['name']: _build_group_transform(group) for group in spec['groups']}


def build_validator_group_transforms(
    validator_protocol: str,
    validator_spec: Optional[Mapping[str, Any]] = None,
    *,
    validator_preset: Optional[str] = None,
) -> Dict[str, transforms.Compose]:
    protocol = str(validator_protocol).lower()
    if protocol == 'handcrafted_va':
        return build_handcrafted_va_group_transforms(
            validator_spec,
            validator_preset=str(validator_preset or HANDCRAFTED_VA_PRESET),
        )
    if protocol == 'llm_va':
        return build_llm_va_group_transforms(validator_spec or {})
    return {}


def build_validator_family_group_transforms(
    validator_family_spec: Mapping[str, Any]
) -> Dict[str, Dict[str, transforms.Compose]]:
    normalized = normalize_validator_family_spec(raw_spec=validator_family_spec)
    payload: Dict[str, Dict[str, transforms.Compose]] = {}
    for member in normalized['validators']:
        payload[member['name']] = build_validator_group_transforms(member['protocol'], member['spec'])
    return payload


def clone_dataset_with_transform(dataset: CustomDataset, transform: Any) -> CustomDataset:
    return CustomDataset(
        img_paths=list(dataset.img_paths),
        transform=transform,
        dataset=dataset.dataset,
        domains=list(dataset.domains) if dataset.domains is not None else None,
        domain_to_idx=dict(dataset.domain_to_idx) if dataset.domain_to_idx is not None else None,
        cached_images=list(dataset.cached_images) if dataset.cached_images is not None else None,
    )


def harmonic_mean(lhs: float, rhs: float) -> float:
    lhs = float(lhs)
    rhs = float(rhs)
    if lhs <= 0.0 or rhs <= 0.0:
        return 0.0
    return 2.0 * lhs * rhs / (lhs + rhs)


def aggregate_validator_metrics(
    validator_protocol: str,
    *,
    vs_acc: float,
    va_group_acc: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    protocol = str(validator_protocol).lower()
    if protocol == 'validator_family':
        raise ValueError('validator_family requires aggregate_validator_family_metrics')
    group_acc = {str(key): float(value) for key, value in (va_group_acc or {}).items()}
    metrics: Dict[str, Any] = {
        'validator_protocol': protocol,
        'selection_metric_name': 'vs_acc' if protocol == 'vs' else 'harmonic_mean_va_avg_and_va_worst',
        'vs_acc': round(float(vs_acc), 4),
        'va_group_acc': {key: round(value, 4) for key, value in group_acc.items()},
        'validator_locked': True,
    }
    if protocol == 'vs':
        metrics['selection_score'] = round(float(vs_acc), 4)
        metrics['va_avg_acc'] = None
        metrics['va_worst_group_acc'] = None
        metrics['va_group_std'] = None
        metrics['va_group_names'] = []
        return metrics

    ordered_names = list(group_acc.keys())
    if not ordered_names:
        raise ValueError(f'{protocol} requires non-empty va_group_acc')
    ordered_values = [group_acc[name] for name in ordered_names]
    va_avg_acc = float(sum(ordered_values) / len(ordered_values))
    va_worst_group_acc = float(min(ordered_values))
    va_group_std = float(np.std(np.asarray(ordered_values, dtype=float), ddof=0))
    selection_score = harmonic_mean(va_avg_acc, va_worst_group_acc)
    metrics['selection_score'] = round(selection_score, 4)
    metrics['va_avg_acc'] = round(va_avg_acc, 4)
    metrics['va_worst_group_acc'] = round(va_worst_group_acc, 4)
    metrics['va_group_std'] = round(va_group_std, 4)
    metrics['va_group_names'] = ordered_names
    return metrics


def stable_hash_payload(payload: Mapping[str, Any]) -> str:
    import json

    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def aggregate_validator_family_metrics(
    validator_family_spec: Mapping[str, Any],
    *,
    vs_acc: float,
    validator_member_scores: Mapping[str, float],
) -> Dict[str, Any]:
    normalized = normalize_validator_family_spec(raw_spec=validator_family_spec)
    scores = {str(name): float(value) for name, value in validator_member_scores.items()}
    expected_names = [member['name'] for member in normalized['validators']]
    if set(scores.keys()) != set(expected_names):
        raise ValueError(
            f'validator_family member scores do not match locked family members: '
            f'expected={expected_names}, got={sorted(scores.keys())}'
        )
    ordered_scores = [scores[name] for name in expected_names]
    va_family_mean = float(sum(ordered_scores) / len(ordered_scores))
    va_family_min = float(min(ordered_scores))
    va_family_max = float(max(ordered_scores))
    va_family_std = float(np.std(np.asarray(ordered_scores, dtype=float), ddof=0))
    family_core = harmonic_mean(va_family_mean, va_family_min)
    return {
        'validator_protocol': 'validator_family',
        'validator_family_protocol': 'validator_family',
        'validator_family_spec': normalized,
        'validator_family_spec_hash': stable_hash_payload(normalized),
        'family_aggregation_mode': FAMILY_AGGREGATION,
        'selection_metric_name': 'q_sr_va_family_hmean',
        'selection_score': round(family_core, 4),
        'vs_acc': round(float(vs_acc), 4),
        'validator_member_scores': {name: round(scores[name], 4) for name in expected_names},
        'va_family_mean': round(va_family_mean, 4),
        'va_family_min': round(va_family_min, 4),
        'va_family_std': round(va_family_std, 4),
        'va_family_max': round(va_family_max, 4),
        'validator_locked': True,
        'va_avg_acc': None,
        'va_worst_group_acc': None,
        'va_group_std': None,
        'va_group_acc': {},
        'va_group_names': [],
    }

