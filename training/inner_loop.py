from __future__ import annotations

import copy
import functools
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader
from torchvision import transforms

import models
from dataset.datasets import build_domain_mappings, concat_load_datasets, extract_all_domains
from training.validation_protocols import (
    FAMILY_AGGREGATION,
    HANDCRAFTED_VA_PRESET,
    LLM_VA_PRESET,
    VALIDATOR_FAMILY_PRESET,
    VALID_VALIDATOR_PROTOCOLS,
    ValidatorSpecError,
    aggregate_validator_family_metrics,
    aggregate_validator_metrics,
    build_standard_val_transform,
    build_validator_family_group_transforms,
    build_validator_group_transforms,
    clone_dataset_with_transform,
    normalize_validator_family_spec,
    normalize_validator_spec,
    stable_hash_payload,
)
from models.qat_layers import QATManager, QuantizedConv2dWrapper, attach_qat_wrappers
from training.schedule import lr_setter
from utils.matrix import accuracy
from utils.meters import AverageMeter


BUDGET_TO_EPOCHS = {
    'short': 6,
    'medium': 12,
    'full': 24,
}
VALID_DISTURB_MODES = {'none', 'rsw', 'qat', 'rsw_qat'}
VALID_BN_MODES = {'train', 'eval'}
VALID_QUANT_SCOPES = {'layer4', 'layer3_4', 'layer2_4'}
VALID_QUANT_BITS = {4, 8}
DATASET_TO_CLASSES = {
    'PACS': 7,
    'VLCS': 5,
    'CAM': 47,
}


class ConfigError(ValueError):
    pass


class TrialFailure(RuntimeError):
    def __init__(self, fail_reason: str, message: str):
        super().__init__(message)
        self.fail_reason = fail_reason


class TrialLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        line = f'[{timestamp}] {message}'
        print(line)
        with open(self.log_path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')


class LinearRandomWeight(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        self.hidden_act = nn.Sigmoid()
        self.linear_out = nn.Linear(dim, 1, bias=True)
        self.output_act = nn.Sigmoid()
        self.weight_init()

    def weight_init(self) -> None:
        torch.nn.init.uniform_(self.linear.weight, a=-1.0, b=1.0)
        torch.nn.init.uniform_(self.linear_out.weight, a=-1.0, b=1.0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, rsw_min: float) -> torch.Tensor:
        self.weight_init()
        weight = self.output_act(self.linear_out(self.hidden_act(self.linear(x)))).clamp(rsw_min, 1.0)
        return weight / weight.mean().clamp_min(1e-6)


@torch.no_grad()
def random_fourier_features_gpu(
    x: torch.Tensor,
    num_f: int,
    w: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)

    if w is None:
        w = torch.randn(size=(num_f, c), device=x.device, dtype=x.dtype)
    if b is None:
        b = 2 * torch.pi * torch.rand(size=(r, num_f), device=x.device, dtype=x.dtype)
    b = b.repeat((n, 1, 1))

    z_scale = torch.sqrt(torch.tensor(2.0 / num_f, device=x.device, dtype=x.dtype))
    mid = torch.matmul(x, w.t())
    mid = mid + b
    mid = mid - mid.min(dim=1, keepdim=True)[0]
    denom = mid.max(dim=1, keepdim=True)[0].clamp_min(1e-6)
    mid = mid / denom
    mid = mid * (torch.pi / 2.0)
    return z_scale * torch.cat((torch.cos(mid), torch.sin(mid), x), dim=-1)


def lossb_expect(
    cfeaturec: torch.Tensor,
    weight: torch.Tensor,
    num_f: int,
    rff_w: Optional[torch.Tensor] = None,
    rff_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, w=rff_w, b=rff_b)
    cfeaturecs = cfeaturecs.permute(2, 0, 1)
    w = weight.view(1, -1, 1)
    wx = cfeaturecs * w
    e = torch.sum(wx, dim=1, keepdim=True)
    res = torch.matmul(wx.transpose(1, 2), cfeaturecs) - torch.matmul(e.transpose(1, 2), e)
    cov_matrix = res * res
    loss = torch.sum(cov_matrix) - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2))
    return loss / cfeaturecs.size(0) / cfeaturecs.size(2)


def _ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise TrialFailure(f'non_finite_{name}', f'{name} contains NaN or Inf')


@torch.amp.autocast('cuda', enabled=False)
def stable_weight_learner(
    cfeatures: torch.Tensor,
    args: SimpleNamespace,
    global_epoch: int = 0,
    iteration: int = 0,
    rsw_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    trace_lossg = os.environ.get('TRACE_STABLE_LOSSG', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    trace_weight = os.environ.get('TRACE_STABLE_WEIGHT', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    trace_input = os.environ.get('TRACE_STABLE_INPUT', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    n = cfeatures.size(0)
    if rsw_weight is None:
        rsw_weight = torch.ones(n, 1, device=cfeatures.device, dtype=torch.float32)

    weight = torch.ones(n, 1, device=cfeatures.device, dtype=torch.float32, requires_grad=True)
    optimizer_bl = torch.optim.SGD([weight], lr=args.lrbl, momentum=args.momentum)

    all_features = cfeatures.detach().float()
    all_rsw_weight = rsw_weight.detach().float()
    rff_w = torch.randn(size=(args.num_f, 1), device=all_features.device, dtype=all_features.dtype)
    rff_b = 2 * torch.pi * torch.rand(size=(all_features.size(1), args.num_f), device=all_features.device, dtype=all_features.dtype)

    for balance_epoch in range(args.epochb):
        lr_setter(optimizer_bl, balance_epoch, args, bl=True)
        optimizer_bl.zero_grad()
        if trace_input:
            with torch.no_grad():
                feature_stats = all_features.detach().float()
                rsw_stats = all_rsw_weight.detach().float()
                weight_stats = weight.detach().float()
                pre_softmax = (weight * all_rsw_weight).detach().float()
                print(
                    '[TRACE] '
                    f'stable_input epoch={global_epoch} iter={iteration} balance_epoch={balance_epoch} '
                    f'feature_mean={feature_stats.mean().item():.6f} feature_std={feature_stats.std(unbiased=False).item():.6f} '
                    f'feature_absmax={feature_stats.abs().max().item():.6f} rsw_mean={rsw_stats.mean().item():.6f} '
                    f'rsw_min={rsw_stats.min().item():.6f} rsw_max={rsw_stats.max().item():.6f} '
                    f'weight_mean={weight_stats.mean().item():.6f} weight_min={weight_stats.min().item():.6f} '
                    f'weight_max={weight_stats.max().item():.6f} pre_softmax_mean={pre_softmax.mean().item():.6f} '
                    f'pre_softmax_min={pre_softmax.min().item():.6f} pre_softmax_max={pre_softmax.max().item():.6f}'
                )
        normalized_weight = F.softmax(weight * all_rsw_weight, dim=0)
        _ensure_finite('stable_weights', normalized_weight)
        lossb = lossb_expect(all_features, normalized_weight, args.num_f, rff_w=rff_w, rff_b=rff_b)
        lossp = normalized_weight.pow(args.decay_pow).sum()
        lossg = lossb / args.lambdap + lossp
        _ensure_finite('lossb', lossb)
        _ensure_finite('lossp', lossp)
        _ensure_finite('lossg', lossg)
        if trace_lossg or trace_weight:
            with torch.no_grad():
                weight_stats = normalized_weight.detach().float()
                raw_stats = (weight * all_rsw_weight).detach().float()
                feature_stats = all_features.detach().float()
                print(
                    '[TRACE] '
                    f'stable_lossg epoch={global_epoch} iter={iteration} balance_epoch={balance_epoch} '
                    f'lossb={lossb.item():.6f} lossp={lossp.item():.6f} lossg={lossg.item():.6f} '
                    f'weight_mean={weight_stats.mean().item():.6f} weight_min={weight_stats.min().item():.6f} '
                    f'weight_max={weight_stats.max().item():.6f} raw_mean={raw_stats.mean().item():.6f} '
                    f'raw_min={raw_stats.min().item():.6f} raw_max={raw_stats.max().item():.6f} '
                    f'feature_mean={feature_stats.mean().item():.6f} feature_std={feature_stats.std(unbiased=False).item():.6f} '
                    f'feature_absmax={feature_stats.abs().max().item():.6f}'
                )
        try:
            lossg.backward()
            if trace_weight:
                grad = weight.grad.detach().float() if weight.grad is not None else None
                if grad is not None:
                    print(
                        '[TRACE] '
                        f'stable_grad epoch={global_epoch} iter={iteration} balance_epoch={balance_epoch} '
                        f'grad_mean={grad.mean().item():.6f} grad_min={grad.min().item():.6f} '
                        f'grad_max={grad.max().item():.6f} grad_norm={grad.norm().item():.6f}'
                    )
            optimizer_bl.step()
            if trace_weight:
                with torch.no_grad():
                    post_weight = weight.detach().float()
                    print(
                        '[TRACE] '
                        f'stable_post_step epoch={global_epoch} iter={iteration} balance_epoch={balance_epoch} '
                        f'weight_mean={post_weight.mean().item():.6f} weight_min={post_weight.min().item():.6f} '
                        f'weight_max={post_weight.max().item():.6f} weight_finite={bool(torch.isfinite(post_weight).all())}'
                    )
        except RuntimeError as exc:
            raise TrialFailure('stable_backward_error', str(exc)) from exc

    learned_weight = (weight * all_rsw_weight[:n]).detach()
    _ensure_finite('learned_weight', learned_weight)
    softmax_weight = F.softmax(learned_weight, dim=0).squeeze(-1)
    _ensure_finite('softmax_weight', softmax_weight)
    if softmax_weight.max() >= 0.1:
        softmax_weight = torch.ones_like(softmax_weight) / n
    return softmax_weight.to(cfeatures.dtype)



def _coerce_bool(value: Any, *, field_name: str) -> bool:
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
    raise ConfigError(f'Invalid boolean value for {field_name}: {value!r}')


def _stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(',', ':'))


def compute_config_hash(config: Dict[str, Any]) -> str:
    excluded = {'runtime_sec', 'log_path', 'trial_dir', 'config_hash'}
    payload = {key: value for key, value in config.items() if key not in excluded}
    return __import__('hashlib').sha256(_stable_json_dumps(payload).encode('utf-8')).hexdigest()


def compute_fallback_config_hash(raw_config: Dict[str, Any]) -> str:
    if not isinstance(raw_config, dict):
        return __import__('hashlib').sha256(repr(raw_config).encode('utf-8')).hexdigest()
    return __import__('hashlib').sha256(_stable_json_dumps(raw_config).encode('utf-8')).hexdigest()


def normalize_config(raw_config: Dict[str, Any], trial_dir: str) -> Dict[str, Any]:
    if not isinstance(raw_config, dict):
        raise ConfigError('Config must be a JSON object')

    def required(name: str) -> Any:
        if name not in raw_config:
            raise ConfigError(f'Missing required config field: {name}')
        return raw_config[name]

    dataset = str(required('dataset')).upper()
    if dataset not in DATASET_TO_CLASSES:
        raise ConfigError(f'Unsupported dataset: {dataset}')

    disturb_mode = str(raw_config.get('disturb_mode', 'rsw')).lower()
    bn_mode = str(required('bn_mode')).lower()
    budget = str(required('budget')).lower()
    validator_protocol = str(raw_config.get('validator_protocol', 'vs')).lower()
    validator_preset_raw = raw_config.get('validator_preset')
    if validator_preset_raw not in {None, ''}:
        validator_preset = str(validator_preset_raw)
    elif validator_protocol == 'handcrafted_va':
        validator_preset = HANDCRAFTED_VA_PRESET
    elif validator_protocol == 'llm_va':
        validator_preset = LLM_VA_PRESET
    elif validator_protocol == 'validator_family':
        validator_preset = VALIDATOR_FAMILY_PRESET
    else:
        validator_preset = 'vs_v0'

    pretrained = raw_config.get('pretrained', True)
    if pretrained is not True:
        raise ConfigError('Inner loop requires pretrained=True; random-initialized ResNet is not supported')

    validator_spec = None
    validator_family_spec = None
    validator_family_spec_hash = None
    try:
        if validator_protocol == 'validator_family':
            validator_family_spec = normalize_validator_family_spec(
                raw_spec=raw_config.get('validator_family_spec'),
                validator_preset=validator_preset,
            )
            validator_family_spec_hash = stable_hash_payload(validator_family_spec)
        else:
            validator_spec = normalize_validator_spec(
                validator_protocol,
                validator_preset=validator_preset,
                raw_spec=raw_config.get('validator_spec'),
            )
    except ValidatorSpecError as exc:
        raise ConfigError(str(exc)) from exc

    config = {
        'dataset': dataset,
        'split_dir': os.path.abspath(str(required('split_dir'))),
        'image_root': os.path.abspath(str(required('image_root'))),
        'gpu': None if raw_config.get('gpu', None) in {None, '', -1} else int(raw_config.get('gpu')),
        'disturb_mode': disturb_mode,
        'bn_mode': bn_mode,
        'budget': budget,
        'validator_protocol': validator_protocol,
        'validator_preset': validator_preset,
        'validator_spec': validator_spec,
        'validator_family_spec': validator_family_spec,
        'validator_family_spec_hash': validator_family_spec_hash,
        'bs': int(required('bs')),
        'lr': float(required('lr')),
        'lambdap': float(required('lambdap')),
        'epochp': int(required('epochp')),
        'seed': int(required('seed')),
        'arch': str(raw_config.get('arch', 'resnet18_with_table')),
        'workers': int(raw_config.get('workers', 0)),
        'prefetch_factor': int(raw_config.get('prefetch_factor', 2)),
        'weight_decay': float(raw_config.get('weight_decay', 1e-4)),
        'print_freq': int(raw_config.get('print_freq', 50)),
        'amp': _coerce_bool(raw_config.get('amp', True), field_name='amp'),
        'pretrained': True,
        'min_scale': float(raw_config.get('min_scale', 0.8)),
        'gray_scale': float(raw_config.get('gray_scale', 0.1)),
        'lrbl': float(raw_config.get('lrbl', 1.0)),
        'epochb': int(raw_config.get('epochb', 20)),
        'decay_pow': float(raw_config.get('decay_pow', 2.0)),
        'rsw_min': float(raw_config.get('rsw_min', 0.2)),
        'num_f': int(required('num_f')),
        'momentum': 0.9,
        'epochs': BUDGET_TO_EPOCHS.get(budget),
        'classes_num': DATASET_TO_CLASSES[dataset],
        'not_dropout': True,
        'backbone_drop': 'none',
        'head_drop': False,
        'drop_rate': 0.0,
        'log_path': str(Path(trial_dir) / 'log.txt'),
        'trial_dir': os.path.abspath(trial_dir),
    }

    if disturb_mode in {'qat', 'rsw_qat'}:
        config['quant_scope'] = str(required('quant_scope'))
        config['quant_bits'] = int(required('quant_bits'))
        config['quant_start_ratio'] = float(required('quant_start_ratio'))
    else:
        config['quant_scope'] = None
        config['quant_bits'] = None
        config['quant_start_ratio'] = None

    config['swa_start_epoch'] = config['epochp']
    config['stable_enabled'] = disturb_mode in {'rsw', 'rsw_qat'}
    config['swa_enabled'] = _coerce_bool(raw_config.get('swa_enabled', False), field_name='swa_enabled')
    config['stable_after_epochp'] = config['stable_enabled']
    config['qat_enabled'] = disturb_mode in {'qat', 'rsw_qat'}
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if config['disturb_mode'] not in VALID_DISTURB_MODES:
        raise ConfigError(f'Unsupported disturb_mode: {config["disturb_mode"]}')
    if config['bn_mode'] not in VALID_BN_MODES:
        raise ConfigError(f'Unsupported bn_mode: {config["bn_mode"]}')
    if config['budget'] not in BUDGET_TO_EPOCHS:
        raise ConfigError(f'Unsupported budget: {config["budget"]}')
    if config['validator_protocol'] not in VALID_VALIDATOR_PROTOCOLS:
        raise ConfigError(f'Unsupported validator_protocol: {config["validator_protocol"]}')
    if config['validator_protocol'] == 'validator_family' and config.get('validator_family_spec') is None:
        raise ConfigError('validator_family runs require a locked validator_family_spec')
    if config['validator_protocol'] != 'validator_family' and config.get('validator_spec') is None and config['validator_protocol'] != 'vs':
        raise ConfigError(f'{config["validator_protocol"]} runs require a locked validator_spec')
    if config['epochs'] is None:
        raise ConfigError(f'Unable to map budget={config["budget"]} to epochs')
    if config['epochp'] >= config['epochs']:
        raise ConfigError(f'Invalid epochp={config["epochp"]} for total epochs={config["epochs"]}')
    if config['bs'] <= 0:
        raise ConfigError('Batch size must be positive')
    if config['workers'] < 0:
        raise ConfigError('workers must be >= 0')
    if config.get('gpu') is not None and int(config['gpu']) < 0:
        raise ConfigError('gpu must be >= 0 or None')
    if config['prefetch_factor'] <= 0:
        raise ConfigError('prefetch_factor must be positive')
    if not os.path.isdir(config['split_dir']):
        raise ConfigError(f'split_dir does not exist: {config["split_dir"]}')
    if not os.path.isdir(config['image_root']):
        raise ConfigError(f'image_root does not exist: {config["image_root"]}')
    if config['qat_enabled']:
        if config['quant_scope'] not in VALID_QUANT_SCOPES:
            raise ConfigError(f'Unsupported quant_scope: {config["quant_scope"]}')
        if config['quant_bits'] not in VALID_QUANT_BITS:
            raise ConfigError(f'Unsupported quant_bits: {config["quant_bits"]}')
        if not (0.0 <= config['quant_start_ratio'] <= 1.0):
            raise ConfigError(f'quant_start_ratio must be in [0, 1], got {config["quant_start_ratio"]}')
    if config['arch'] not in models.__dict__:
        raise ConfigError(f'Unsupported arch: {config["arch"]}')


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def build_runtime_args(config: Dict[str, Any], trial_dir: str) -> SimpleNamespace:
    args = SimpleNamespace(**config)
    args.data = config['split_dir']
    args.image_root = config['image_root']
    args.batch_size = config['bs']
    args.evaluate = False
    args.resume = ''
    args.start_epoch = 0
    args.cos = 1
    args.distributed = False
    args.world_size = 1
    args.multiprocessing_distributed = False
    args.dist_url = ''
    args.dist_backend = 'nccl'
    args.rank = 0
    args.gpu = int(config['gpu']) if config.get('gpu') is not None and torch.cuda.is_available() else (0 if torch.cuda.is_available() else None)
    args.trial_dir = trial_dir
    args.config_hash = config.get('config_hash')
    return args


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloaders(args: SimpleNamespace):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.min_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4, .4, .4, .4),
        transforms.RandomGrayscale(args.gray_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = build_standard_val_transform()

    all_domains = extract_all_domains(args.data, data_types=['train', 'val', 'test'])
    domain_to_idx, idx_to_domain = build_domain_mappings(all_domains)

    train_dataset = concat_load_datasets(args.data, args.image_root, 'train', train_transform, args.dataset, domain_to_idx)
    val_dataset = concat_load_datasets(args.data, args.image_root, 'val', val_transform, args.dataset, domain_to_idx)
    test_dataset = concat_load_datasets(args.data, args.image_root, 'test', val_transform, args.dataset, domain_to_idx)

    if train_dataset is None or val_dataset is None or test_dataset is None:
        raise ConfigError('Failed to build one or more datasets from split_dir/image_root')

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common_loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'pin_memory': args.gpu is not None,
        'persistent_workers': args.workers > 0,
        'worker_init_fn': functools.partial(_seed_worker, base_seed=args.seed),
        'generator': generator,
    }
    if args.workers > 0:
        common_loader_kwargs['prefetch_factor'] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common_loader_kwargs)

    validator_loaders = {
        'vs': val_loader,
        'va_groups': {},
        'validator_spec': args.validator_spec,
        'validator_family_spec': getattr(args, 'validator_family_spec', None),
        'family_members': {},
    }
    if args.validator_protocol == 'validator_family':
        family_transforms = build_validator_family_group_transforms(args.validator_family_spec)
        for member in args.validator_family_spec['validators']:
            member_name = member['name']
            validator_loaders['family_members'][member_name] = {
                'protocol': member['protocol'],
                'spec': member['spec'],
                'va_groups': {},
            }
            for group_name, group_transform in family_transforms.get(member_name, {}).items():
                group_dataset = clone_dataset_with_transform(val_dataset, group_transform)
                validator_loaders['family_members'][member_name]['va_groups'][group_name] = DataLoader(
                    group_dataset,
                    shuffle=False,
                    drop_last=False,
                    **common_loader_kwargs,
                )
    else:
        for group_name, group_transform in build_validator_group_transforms(args.validator_protocol, args.validator_spec).items():
            group_dataset = clone_dataset_with_transform(val_dataset, group_transform)
            validator_loaders['va_groups'][group_name] = DataLoader(
                group_dataset,
                shuffle=False,
                drop_last=False,
                **common_loader_kwargs,
            )
    return train_loader, val_loader, test_loader, validator_loaders, all_domains, idx_to_domain


def build_model(args: SimpleNamespace, logger: TrialLogger):
    model_ctor = models.__dict__[args.arch]
    model_init_args = SimpleNamespace(
        head_drop=False,
        backbone_drop='none',
        drop_rate=0.0,
        not_dropout=True,
        batch_size=args.batch_size,
    )
    model = model_ctor(pretrained=args.pretrained, args=model_init_args)
    num_ftrs = model.fc1.in_features
    model.fc1 = nn.Linear(num_ftrs, args.classes_num)
    nn.init.xavier_uniform_(model.fc1.weight, 0.1)
    nn.init.constant_(model.fc1.bias, 0.0)

    qat_manager = None
    if args.qat_enabled:
        qat_manager = attach_qat_wrappers(model, args.quant_scope, args.quant_bits)
        logger.log(
            f'Attached QAT wrappers: scope={args.quant_scope}, bits={args.quant_bits}, wrapped_convs={qat_manager.num_wrapped_convs()}'
        )

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    return model, qat_manager


def set_bn_mode(model: nn.Module, mode: str) -> None:
    if mode == 'train':
        return
    if mode != 'eval':
        raise ValueError(f'Unsupported BN mode: {mode}')
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _compute_device(args: SimpleNamespace) -> torch.device:
    if args.gpu is not None:
        return torch.device(f'cuda:{args.gpu}')
    return torch.device('cpu')


def _enable_qat_wrappers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, QuantizedConv2dWrapper):
            module.set_enabled(True)



def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: SimpleNamespace,
    frsw: LinearRandomWeight,
    scaler: Optional[GradScaler],
    logger: TrialLogger,
    qat_manager: Optional[QATManager],
) -> Dict[str, float]:
    loss_meter = AverageMeter('Loss', ':6.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()
    set_bn_mode(model, args.bn_mode)
    steps_per_epoch = len(train_loader)

    for batch_idx, (images, target, _) in enumerate(train_loader):
        global_step = epoch * steps_per_epoch + batch_idx
        if qat_manager is not None:
            enabled_now = qat_manager.maybe_enable(
                global_step=global_step,
                total_steps=args.epochs * steps_per_epoch,
                start_ratio=args.quant_start_ratio,
            )
            if enabled_now:
                logger.log(
                    f'QAT enabled at global_step={global_step} ({global_step / max(args.epochs * steps_per_epoch, 1):.4f})'
                )

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=bool(args.amp and args.gpu is not None)):
            output, cfeatures = model(images)
            cfeatures = cfeatures.float()
            if args.stable_enabled and epoch >= args.epochp and cfeatures.size(0) > 16:
                rsw_weight = frsw(cfeatures, args.rsw_min)
                sample_weight = stable_weight_learner(cfeatures, args, epoch, batch_idx, rsw_weight)
            else:
                sample_weight = cfeatures.new_ones(cfeatures.size(0)) / cfeatures.size(0)
            _ensure_finite('sample_weight', sample_weight)
            loss_vec = F.cross_entropy(output, target, reduction='none')
            loss = torch.sum(loss_vec * sample_weight)
        _ensure_finite('train_loss', loss.detach().float())

        try:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        except RuntimeError as exc:
            raise TrialFailure('optimizer_step_error', str(exc)) from exc

        batch_size = target.size(0)
        acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
        loss_meter.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        if batch_idx % args.print_freq == 0:
            logger.log(
                f'Epoch {epoch:02d} Iter {batch_idx:04d}/{steps_per_epoch} '
                f'Loss={loss_meter.avg:.4f} Acc@1={top1.avg:.2f} Acc@5={top5.avg:.2f}'
            )

    return {'loss': loss_meter.avg, 'acc1': top1.avg, 'acc5': top5.avg}


def _prepare_eval_model(
    model: nn.Module,
    swa_model: Optional[AveragedModel],
    train_loader: DataLoader,
    args: SimpleNamespace,
    use_swa: bool,
) -> nn.Module:
    if not use_swa:
        return model
    if swa_model is None:
        raise TrialFailure('missing_swa_model', 'SWA evaluation requested but no SWA model was initialized')
    eval_model = swa_model.module
    if args.bn_mode == 'train':
        update_bn(train_loader, eval_model, device=_compute_device(args))
    return eval_model


def evaluate_model(
    data_loader: DataLoader,
    eval_model: nn.Module,
    args: SimpleNamespace,
    logger: TrialLogger,
    split_name: str,
    epoch: int,
    all_domains=None,
    separate_domain: bool = False,
) -> Tuple[float, Dict[str, float]]:
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    domain_top1: Dict[str, AverageMeter] = {}

    eval_model.eval()
    with torch.no_grad():
        for images, target, domains in data_loader:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                domains = domains.cuda(args.gpu, non_blocking=True)

            with autocast('cuda', enabled=bool(args.amp and args.gpu is not None)):
                output, _ = eval_model(images)

            batch_size = target.size(0)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            if separate_domain and all_domains is not None:
                unique_domains = domains.unique()
                for domain_idx in unique_domains:
                    domain_int = int(domain_idx.item())
                    if domain_int < 0 or domain_int >= len(all_domains):
                        continue
                    mask = domains == domain_idx
                    if int(mask.sum().item()) == 0:
                        continue
                    acc1_d, _ = accuracy(output[mask], target[mask], topk=(1, 5))
                    domain_name = all_domains[domain_int]
                    meter = domain_top1.setdefault(domain_name, AverageMeter('Acc@1', ':6.2f'))
                    meter.update(acc1_d.item(), int(mask.sum().item()))

    logger.log(f'{split_name} epoch={epoch} Acc@1={top1.avg:.3f} Acc@5={top5.avg:.3f}')
    domain_summary = {name: round(meter.avg, 2) for name, meter in domain_top1.items()}
    if domain_summary:
        logger.log(f'{split_name} domain Acc@1: {json.dumps(domain_summary, ensure_ascii=False, sort_keys=True)}')
    return top1.avg, domain_summary


def evaluate_validator_protocol(
    validator_loaders: Dict[str, Any],
    eval_model: nn.Module,
    args: SimpleNamespace,
    logger: TrialLogger,
    epoch: int,
) -> Dict[str, Any]:
    vs_acc, _ = evaluate_model(validator_loaders['vs'], eval_model, args, logger, 'vs_val', epoch)
    if args.validator_protocol == 'validator_family':
        validator_member_scores: Dict[str, float] = {}
        for member_name, member_payload in (validator_loaders.get('family_members') or {}).items():
            va_group_acc: Dict[str, float] = {}
            for group_name, data_loader in (member_payload.get('va_groups') or {}).items():
                group_acc, _ = evaluate_model(data_loader, eval_model, args, logger, f'family_{member_name}_{group_name}', epoch)
                va_group_acc[group_name] = group_acc
            member_metrics = aggregate_validator_metrics(
                member_payload['protocol'],
                vs_acc=vs_acc,
                va_group_acc=va_group_acc,
            )
            validator_member_scores[member_name] = float(member_metrics['selection_score'])
        metrics = aggregate_validator_family_metrics(
            args.validator_family_spec,
            vs_acc=vs_acc,
            validator_member_scores=validator_member_scores,
        )
    else:
        va_group_acc: Dict[str, float] = {}
        for group_name, data_loader in (validator_loaders.get('va_groups') or {}).items():
            group_acc, _ = evaluate_model(data_loader, eval_model, args, logger, f'va_{group_name}', epoch)
            va_group_acc[group_name] = group_acc
        metrics = aggregate_validator_metrics(
            args.validator_protocol,
            vs_acc=vs_acc,
            va_group_acc=va_group_acc,
        )
    logger.log(f'Validator metrics epoch={epoch}: {json.dumps(metrics, ensure_ascii=False, sort_keys=True)}')
    return metrics


def _checkpoint_payload(
    state_dict: Dict[str, torch.Tensor],
    epoch: int,
    best_val_acc1: float,
    selection_score: float,
    args: SimpleNamespace,
    validator_metrics: Dict[str, Any],
    *,
    epoch_val_acc1: float | None = None,
    epoch_selection_score: float | None = None,
) -> Dict[str, Any]:
    return {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': state_dict,
        'best_val_acc1': best_val_acc1,
        'selection_score': selection_score,
        'epoch_val_acc1': epoch_val_acc1,
        'epoch_selection_score': epoch_selection_score,
        'selection_metric_name': validator_metrics.get('selection_metric_name'),
        'validator_protocol': getattr(args, 'validator_protocol', 'vs'),
        'validator_family_protocol': 'validator_family' if getattr(args, 'validator_protocol', 'vs') == 'validator_family' else None,
        'validator_family_spec_hash': getattr(args, 'validator_family_spec_hash', None),
        'config_hash': args.config_hash,
        'disturb_mode': args.disturb_mode,
        'swa_enabled': args.swa_enabled,
        'stable_enabled': args.stable_enabled,
        'seed': args.seed,
    }


def run_inner_loop(config: Dict[str, Any], trial_dir: str) -> Dict[str, Any]:
    args = build_runtime_args(config, trial_dir)
    logger = TrialLogger(args.log_path)
    logger.log(f'Starting trial config_hash={args.config_hash}')
    logger.log(f'Config summary: {json.dumps(config, sort_keys=True, ensure_ascii=False)}')

    set_global_determinism(args.seed)
    model, qat_manager = build_model(args, logger)
    train_loader, val_loader, test_loader, validator_loaders, all_domains, _ = build_dataloaders(args)
    logger.log(
        f'Dataloader config: workers={args.workers} persistent_workers={args.workers > 0} ' 
        f'prefetch_factor={args.prefetch_factor if args.workers > 0 else None} pin_memory={args.gpu is not None}'
    )
    if args.validator_protocol == 'validator_family':
        validator_descriptor = {
            'protocol': args.validator_protocol,
            'preset': args.validator_preset,
            'family_aggregation': FAMILY_AGGREGATION,
            'members': [member['name'] for member in (args.validator_family_spec or {}).get('validators', [])],
            'include_vs': bool((args.validator_family_spec or {}).get('include_vs', True)),
        }
    else:
        validator_descriptor = {
            'protocol': args.validator_protocol,
            'preset': args.validator_preset,
            'groups': [group.get('name') for group in (args.validator_spec or {}).get('groups', [])],
        }
    logger.log(f'Validator protocol: {json.dumps(validator_descriptor, ensure_ascii=False, sort_keys=True)}')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler('cuda', enabled=bool(args.amp and args.gpu is not None))
    frsw = LinearRandomWeight(model.fc1.in_features)
    if args.gpu is not None:
        frsw = frsw.cuda(args.gpu)

    swa_model: Optional[AveragedModel] = None
    if args.swa_enabled:
        swa_model = AveragedModel(model)
        if args.gpu is not None:
            swa_model = swa_model.cuda(args.gpu)

    best_selection_score = float('-inf')
    best_val_acc1 = float('-inf')
    best_val_epoch = None
    best_validator_metrics: Optional[Dict[str, Any]] = None
    best_checkpoint_path = Path(trial_dir) / 'model_best_val.pth'
    epoch_checkpoint_dir = Path(trial_dir) / 'epoch_checkpoints'
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_checkpoint_rows: list[Dict[str, Any]] = []
    started_swa = False

    for epoch in range(args.epochs):
        epoch_wall_start = time.perf_counter()
        lr_setter(optimizer, epoch, args)

        train_wall_start = time.perf_counter()
        train_metrics = train_one_epoch(train_loader, model, optimizer, epoch, args, frsw, scaler, logger, qat_manager)
        train_wall_sec = time.perf_counter() - train_wall_start
        logger.log(f'Train epoch={epoch} summary: {json.dumps(train_metrics, sort_keys=True)} train_wall_sec={train_wall_sec:.3f}')

        use_swa = bool(args.swa_enabled and epoch >= args.swa_start_epoch)
        if use_swa:
            if swa_model is None:
                raise TrialFailure('missing_swa_model', 'SWA was enabled but no SWA model exists')
            if args.qat_enabled and qat_manager is not None and qat_manager.is_enabled():
                _enable_qat_wrappers(swa_model.module)
            swa_model.update_parameters(model)
            started_swa = True
        prep_wall_start = time.perf_counter()
        eval_model = _prepare_eval_model(model, swa_model, train_loader, args, use_swa=use_swa)
        prep_wall_sec = time.perf_counter() - prep_wall_start
        val_wall_start = time.perf_counter()
        validator_metrics = evaluate_validator_protocol(validator_loaders, eval_model, args, logger, epoch)
        val_wall_sec = time.perf_counter() - val_wall_start
        selection_score = float(validator_metrics['selection_score'])
        logger.log(
            f'Epoch {epoch} eval prep_wall_sec={prep_wall_sec:.3f} val_wall_sec={val_wall_sec:.3f} '
            f'selection_score={selection_score:.3f} use_swa={use_swa}'
        )
        allow_best_update = use_swa or not args.swa_enabled
        if allow_best_update and selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_acc1 = float(validator_metrics['vs_acc'])
            best_validator_metrics = dict(validator_metrics)
            best_val_epoch = epoch
            torch.save(
                _checkpoint_payload(
                    copy.deepcopy(eval_model.state_dict()),
                    epoch,
                    best_val_acc1,
                    best_selection_score,
                    args,
                    best_validator_metrics,
                ),
                best_checkpoint_path,
            )
            logger.log(
                f'Updated best checkpoint at epoch={epoch} selection_score={best_selection_score:.3f} '
                f'use_swa={use_swa}'
            )

        epoch_wall_sec = time.perf_counter() - epoch_wall_start
        epoch_checkpoint_path = epoch_checkpoint_dir / f'epoch_{epoch:03d}.pth'
        torch.save(
            _checkpoint_payload(
                copy.deepcopy(eval_model.state_dict()),
                epoch,
                best_val_acc1 if best_val_acc1 != float('-inf') else float(validator_metrics['vs_acc']),
                selection_score,
                args,
                validator_metrics,
                epoch_val_acc1=float(validator_metrics['vs_acc']),
                epoch_selection_score=selection_score,
            ),
            epoch_checkpoint_path,
        )
        epoch_checkpoint_rows.append(
            {
                'epoch': epoch,
                'epoch_checkpoint_path': str(epoch_checkpoint_path),
                'selection_score': round(selection_score, 6),
                'vs_acc': round(float(validator_metrics['vs_acc']), 6),
                'train_loss': round(float(train_metrics['loss']), 6),
                'train_acc1': round(float(train_metrics['acc1']), 6),
            }
        )
        logger.log(f'Epoch {epoch} total_wall_sec={epoch_wall_sec:.3f}')

    if args.swa_enabled:
        if not started_swa or best_val_epoch is None or best_validator_metrics is None or not best_checkpoint_path.exists():
            raise TrialFailure('missing_swa_checkpoint', 'SWA never started or no best SWA checkpoint was produced')
    elif best_val_epoch is None or best_validator_metrics is None or not best_checkpoint_path.exists():
        raise TrialFailure('missing_best_checkpoint', 'No best checkpoint was produced for non-SWA run')

    best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
    model.load_state_dict(best_checkpoint['state_dict'])
    if args.gpu is not None:
        model = model.cuda(args.gpu)

    test_wall_start = time.perf_counter()
    best_test_acc1, domain_acc = evaluate_model(
        test_loader,
        model,
        args,
        logger,
        'test',
        best_val_epoch,
        all_domains=all_domains,
        separate_domain=True,
    )
    logger.log(f'Final test wall_sec={time.perf_counter() - test_wall_start:.3f}')

    result = {
        'status': 'ok',
        'config_hash': args.config_hash,
        'dataset': args.dataset,
        'split_dir': args.data,
        'image_root': args.image_root,
        'seed': args.seed,
        'disturb_mode': args.disturb_mode,
        'swa_enabled': args.swa_enabled,
        'stable_enabled': args.stable_enabled,
        'quant_scope': args.quant_scope,
        'quant_bits': args.quant_bits,
        'quant_start_ratio': args.quant_start_ratio,
        'bn_mode': args.bn_mode,
        'validator_protocol': args.validator_protocol,
        'validator_preset': args.validator_preset,
        'validator_spec': copy.deepcopy(args.validator_spec),
        'validator_family_protocol': 'validator_family' if args.validator_protocol == 'validator_family' else None,
        'validator_family_spec': copy.deepcopy(args.validator_family_spec),
        'validator_family_spec_hash': args.validator_family_spec_hash,
        'family_aggregation_mode': FAMILY_AGGREGATION if args.validator_protocol == 'validator_family' else None,
        'selection_metric_name': best_validator_metrics['selection_metric_name'],
        'selection_score': round(float(best_validator_metrics['selection_score']), 4),
        'vs_acc': round(float(best_validator_metrics['vs_acc']), 4),
        'va_avg_acc': best_validator_metrics['va_avg_acc'],
        'va_worst_group_acc': best_validator_metrics['va_worst_group_acc'],
        'va_group_std': best_validator_metrics['va_group_std'],
        'va_group_acc': dict(best_validator_metrics.get('va_group_acc') or {}),
        'validator_member_scores': dict(best_validator_metrics.get('validator_member_scores') or {}),
        'va_family_mean': best_validator_metrics.get('va_family_mean'),
        'va_family_min': best_validator_metrics.get('va_family_min'),
        'va_family_std': best_validator_metrics.get('va_family_std'),
        'va_family_max': best_validator_metrics.get('va_family_max'),
        'bs': args.batch_size,
        'lr': args.lr,
        'lambdap': args.lambdap,
        'epochp': args.epochp,
        'weight_decay': args.weight_decay,
        'budget': args.budget,
        'epochs': args.epochs,
        'best_val_acc1': round(best_val_acc1, 4),
        'best_test_acc1': round(best_test_acc1, 4),
        'best_val_epoch': best_val_epoch,
        'best_selection_epoch': best_val_epoch,
        'checkpoint_type': 'swa_epoch_checkpoint' if args.swa_enabled else 'base_epoch_checkpoint',
    }
    if domain_acc:
        result['domain_acc'] = domain_acc
    result['epoch_checkpoint_dir'] = str(epoch_checkpoint_dir)
    result['epoch_checkpoint_count'] = len(epoch_checkpoint_rows)
    return result



