from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from dataset.datasets import maybe_enable_dataset_image_cache
from utils.io import read_json
from torchvision.transforms import functional as TF

VIEW_NAMES = [
    "source_val",
    "color_jitter_low",
    "color_jitter_medium",
    "gaussian_blur_low",
    "gaussian_blur_medium",
    "grayscale",
    "noise_low",
    "random_resized_crop_mild",
    "autocontrast",
    "sharpness",
    "posterize",
    "solarize",
]


def _phase2b_log(message: str) -> None:
    print(f"[Phase II-B] {message}", flush=True)

_CHECKPOINT_CACHE_LOCK = threading.Lock()
_CHECKPOINT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CHECKPOINT_CACHE_BYTES = 0


def _checkpoint_cache_enabled() -> bool:
    return str(os.environ.get("PHASE2B_CHECKPOINT_CACHE", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}


def _checkpoint_cache_max_bytes() -> int:
    raw = os.environ.get("PHASE2B_CHECKPOINT_CACHE_MAX_GB", "96")
    if raw is None:
        return 96 * 1024**3
    cleaned = str(raw).strip().lower()
    if cleaned in {"", "none", "null"}:
        return 96 * 1024**3
    return int(float(cleaned) * (1024**3))


def _state_dict_size_bytes(state_dict: Mapping[str, Any]) -> int:
    import torch

    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.element_size()) * int(value.numel())
    return total


def resolve_phase2b_gpu_devices() -> list[int | None]:
    import torch

    if not torch.cuda.is_available():
        return [None]
    raw = os.environ.get("VAL_EVAL_GPU_DEVICES", "0 1")
    devices: list[int] = []
    for token in str(raw).split():
        try:
            devices.append(int(token))
        except ValueError:
            continue
    return devices or [0]


def clear_checkpoint_cache() -> None:
    global _CHECKPOINT_CACHE_BYTES
    with _CHECKPOINT_CACHE_LOCK:
        _CHECKPOINT_CACHE.clear()
        _CHECKPOINT_CACHE_BYTES = 0


def load_checkpoint_bundle(checkpoint_path: Path) -> dict[str, Any]:
    global _CHECKPOINT_CACHE_BYTES
    import torch

    checkpoint_path = Path(checkpoint_path).resolve()
    cache_key = str(checkpoint_path)
    if _checkpoint_cache_enabled():
        with _CHECKPOINT_CACHE_LOCK:
            cached = _CHECKPOINT_CACHE.get(cache_key)
            if cached is not None:
                _CHECKPOINT_CACHE.move_to_end(cache_key)
                return cached

    payload = torch.load(checkpoint_path, map_location="cpu")
    bundle = {
        "epoch": int(payload.get("epoch", 0)),
        "state_dict": payload["state_dict"],
    }

    if _checkpoint_cache_enabled():
        size_bytes = _state_dict_size_bytes(bundle["state_dict"])
        max_bytes = _checkpoint_cache_max_bytes()
        with _CHECKPOINT_CACHE_LOCK:
            if size_bytes <= max_bytes:
                _CHECKPOINT_CACHE[cache_key] = bundle
                _CHECKPOINT_CACHE.move_to_end(cache_key)
                _CHECKPOINT_CACHE_BYTES += size_bytes
                while _CHECKPOINT_CACHE and _CHECKPOINT_CACHE_BYTES > max_bytes:
                    _, evicted = _CHECKPOINT_CACHE.popitem(last=False)
                    _CHECKPOINT_CACHE_BYTES -= _state_dict_size_bytes(evicted["state_dict"])
    return bundle


class _NullLogger:
    def log(self, message: str) -> None:
        del message


def _noise_transform(std: float, *, seed: int) -> transforms.Compose:
    import torch
    from torchvision import transforms

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    class _GaussianNoise:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype) * float(std)
            return (tensor + noise).clamp(0.0, 1.0)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _GaussianNoise(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _view_transform(view: str, *, seed: int) -> transforms.Compose:
    from torchvision import transforms

    if view == "source_val":
        from training.validation_protocols import build_standard_val_transform

        return build_standard_val_transform()
    if view == "color_jitter_low":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "color_jitter_medium":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "gaussian_blur_low":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=5, sigma=0.7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "gaussian_blur_medium":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=7, sigma=1.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "grayscale":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "noise_low":
        return _noise_transform(0.02, seed=seed)
    if view == "random_resized_crop_mild":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.92, 1.0), ratio=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "autocontrast":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(TF.autocontrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "sharpness":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image: TF.adjust_sharpness(image, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "posterize":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image: TF.posterize(image, bits=6)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if view == "solarize":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image: TF.solarize(image, threshold=128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    raise ValueError(f"Unsupported validation view: {view}")


def build_transform_from_view_spec(view: Mapping[str, Any], *, seed: int):
    from torchvision import transforms
    operator = str(view.get("operator") or "identity")
    params = dict(view.get("params") or {})
    if operator == "identity":
        from training.validation_protocols import build_standard_val_transform

        return build_standard_val_transform()
    if operator == "color_jitter":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(
                brightness=float(params.get("brightness", 0.1)),
                contrast=float(params.get("contrast", 0.1)),
                saturation=float(params.get("saturation", 0.1)),
                hue=float(params.get("hue", 0.02)),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if operator == "gaussian_blur":
        sigma = float(params.get("sigma", 0.5))
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=5 if sigma <= 0.5 else 7, sigma=sigma),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if operator == "grayscale":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if operator == "noise":
        return _noise_transform(float(params.get("std", 0.02)), seed=seed)
    if operator == "random_resized_crop":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(float(params.get("scale_min", 0.9)), float(params.get("scale_max", 1.0))),
                ratio=(float(params.get("ratio_min", 0.95)), float(params.get("ratio_max", 1.05))),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if operator == "autocontrast":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(TF.autocontrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if operator == "sharpness":
        factor = float(params.get("factor", 1.0))
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image: TF.adjust_sharpness(image, factor)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if operator == "posterize":
        bits = int(round(float(params.get("bits", 6))))
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image: TF.posterize(image, bits=max(1, min(8, bits)))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if operator == "solarize":
        threshold = float(params.get("threshold", 128))
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image: TF.solarize(image, threshold=threshold)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    raise ValueError(f"Unsupported validation view operator: {operator}")


def _trial_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "epoch_checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _enable_phase2b_image_cache(dataset, *, label: str) -> dict[str, Any]:
    enabled = str(os.environ.get("PHASE2B_IMAGE_CACHE", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
    max_gb_raw = os.environ.get("PHASE2B_IMAGE_CACHE_MAX_GB", "96")
    max_gb = None if max_gb_raw is None or str(max_gb_raw).strip().lower() in {"", "none", "null"} else float(max_gb_raw)
    if not enabled:
        return {
            "enabled": False,
            "label": label,
            "count": len(dataset) if dataset is not None else 0,
            "cache_bytes": 0,
            "cache_gb": 0.0,
            "reason": "disabled",
        }
    return maybe_enable_dataset_image_cache(dataset, enabled=True, max_gb=max_gb, label=label)


def _checkpoint_row(checkpoint_path: Path, *, status: str, dataset: str, split_name: str, seed: int, epoch: int, checkpoint_payload: Mapping[str, Any] | None = None, error: Exception | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "checkpoint_id": checkpoint_path.stem,
        "dataset": dataset,
        "split": split_name,
        "seed": int(seed),
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "status": status,
        "fail_reason": None,
        "error_message": None,
        "source_val": None,
        "color_jitter_low": None,
        "color_jitter_medium": None,
        "gaussian_blur_low": None,
        "gaussian_blur_medium": None,
        "grayscale": None,
        "noise_low": None,
        "random_resized_crop_mild": None,
        "test_acc": None,
        "selection_anchor": None,
    }
    if checkpoint_payload is not None:
        row["selection_anchor"] = float(checkpoint_payload.get("epoch_selection_score") or checkpoint_payload.get("selection_score") or 0.0)
    if error is not None:
        row["fail_reason"] = error.__class__.__name__
        row["error_message"] = str(error)
    return row


def evaluate_checkpoint_scores(checkpoint_path: Path, *, dataset: str, gpu_override: int | None = None) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from training.inner_loop import build_dataloaders, build_model, build_runtime_args, evaluate_model, set_global_determinism
    from training.validation_protocols import clone_dataset_with_transform

    checkpoint_path = Path(checkpoint_path)
    trial_dir = _trial_dir_from_checkpoint(checkpoint_path)
    config_path = trial_dir / "config.json"
    result_path = trial_dir / "result.json"
    if not config_path.exists() or not result_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata for {checkpoint_path}")

    config = read_json(config_path)
    result = read_json(result_path)
    seed = int(config.get("seed", 0))
    split_dir = str(config.get("split_dir") or "")
    split_name = Path(split_dir).name or "unknown_split"
    checkpoint_payload = load_checkpoint_bundle(checkpoint_path)
    epoch = int(checkpoint_payload.get("epoch", 0))

    runtime_config = dict(config)
    runtime_config["workers"] = 0
    runtime_config["gpu"] = int(gpu_override) if gpu_override is not None and torch.cuda.is_available() else (None if not torch.cuda.is_available() else runtime_config.get("gpu"))
    runtime_args = build_runtime_args(runtime_config, str(trial_dir))
    runtime_args.workers = 0
    runtime_args.prefetch_factor = 2
    runtime_args.batch_size = int(runtime_args.batch_size)
    runtime_args.gpu = runtime_args.gpu if torch.cuda.is_available() else None
    set_global_determinism(seed)
    logger = _NullLogger()
    model, _ = build_model(runtime_args, logger)  # type: ignore[arg-type]

    train_loader, val_loader, test_loader, _, _, _ = build_dataloaders(runtime_args)
    del train_loader
    _enable_phase2b_image_cache(val_loader.dataset, label=f"{dataset.upper()}_checkpoint_val")
    _enable_phase2b_image_cache(test_loader.dataset, label=f"{dataset.upper()}_checkpoint_test")

    model.load_state_dict(checkpoint_payload["state_dict"])
    if runtime_args.gpu is not None:
        model = model.cuda(runtime_args.gpu)

    row = _checkpoint_row(
        checkpoint_path,
        status="ok",
        dataset=dataset.upper(),
        split_name=split_name,
        seed=seed,
        epoch=epoch,
        checkpoint_payload=checkpoint_payload,
    )

    try:
        source_val, _ = evaluate_model(val_loader, model, runtime_args, logger, "source_val", epoch)
        row["source_val"] = round(float(source_val), 6)
        for offset, view_name in enumerate([name for name in VIEW_NAMES if name != "source_val"], start=1):
            set_global_determinism(seed + epoch + offset)
            view_dataset = clone_dataset_with_transform(val_loader.dataset, _view_transform(view_name, seed=seed + epoch + offset))
            view_loader = DataLoader(
                view_dataset,
                batch_size=int(runtime_args.batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=runtime_args.gpu is not None,
            )
            score, _ = evaluate_model(view_loader, model, runtime_args, logger, view_name, epoch)
            row[view_name] = round(float(score), 6)
        test_acc, _ = evaluate_model(test_loader, model, runtime_args, logger, "test", epoch)
        row["test_acc"] = round(float(test_acc), 6)
    except Exception as exc:
        row["status"] = "fail"
        row["fail_reason"] = exc.__class__.__name__
        row["error_message"] = str(exc)

    row["checkpoint_path"] = str(checkpoint_path)
    row["dataset"] = dataset.upper()
    return row


def evaluate_checkpoint_view_score(checkpoint_path: Path, *, dataset: str, view: Mapping[str, Any], gpu_override: int | None = None) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from training.inner_loop import build_dataloaders, build_model, build_runtime_args, evaluate_model, set_global_determinism
    from training.validation_protocols import clone_dataset_with_transform

    checkpoint_path = Path(checkpoint_path)
    trial_dir = _trial_dir_from_checkpoint(checkpoint_path)
    config = read_json(trial_dir / "config.json")
    seed = int(config.get("seed", 0))
    split_dir = str(config.get("split_dir") or "")
    split_name = Path(split_dir).name or "unknown_split"
    checkpoint_payload = load_checkpoint_bundle(checkpoint_path)
    epoch = int(checkpoint_payload.get("epoch", 0))
    runtime_config = dict(config)
    runtime_config["workers"] = 0
    runtime_config["gpu"] = int(gpu_override) if gpu_override is not None and torch.cuda.is_available() else (None if not torch.cuda.is_available() else runtime_config.get("gpu"))
    runtime_args = build_runtime_args(runtime_config, str(trial_dir))
    runtime_args.workers = 0
    runtime_args.prefetch_factor = 2
    runtime_args.gpu = runtime_args.gpu if torch.cuda.is_available() else None
    set_global_determinism(seed + epoch)
    logger = _NullLogger()
    model, _ = build_model(runtime_args, logger)  # type: ignore[arg-type]
    train_loader, val_loader, _, _, _, _ = build_dataloaders(runtime_args)
    del train_loader
    _enable_phase2b_image_cache(val_loader.dataset, label=f"{dataset.upper()}_checkpoint_val")
    model.load_state_dict(checkpoint_payload["state_dict"])
    if runtime_args.gpu is not None:
        model = model.cuda(runtime_args.gpu)
    view_transform = build_transform_from_view_spec(view, seed=seed + epoch)
    view_dataset = clone_dataset_with_transform(val_loader.dataset, view_transform)
    view_loader = DataLoader(
        view_dataset,
        batch_size=int(runtime_args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=runtime_args.gpu is not None,
    )
    score, _ = evaluate_model(view_loader, model, runtime_args, logger, str(view.get("name") or view.get("signature") or "view"), epoch)
    return {
        "dataset": dataset.upper(),
        "split": split_name,
        "seed": seed,
        "epoch": epoch,
        "checkpoint_id": checkpoint_path.stem,
        "view_name": view.get("name"),
        "view_signature": view.get("signature"),
        "val_score": round(float(score), 6),
        "status": "ok",
        "checkpoint_path": str(checkpoint_path),
    }


def evaluate_checkpoint_group_view_scores(checkpoint_paths: Sequence[Path], *, dataset: str, view: Mapping[str, Any], gpu_override: int | None = None) -> list[Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from training.inner_loop import build_dataloaders, build_model, build_runtime_args, evaluate_model, set_global_determinism
    from training.validation_protocols import clone_dataset_with_transform

    checkpoint_paths = [Path(path) for path in checkpoint_paths]
    if not checkpoint_paths:
        return []
    first_trial_dir = _trial_dir_from_checkpoint(checkpoint_paths[0])
    config = read_json(first_trial_dir / "config.json")
    seed = int(config.get("seed", 0))
    split_dir = str(config.get("split_dir") or "")
    split_name = Path(split_dir).name or "unknown_split"
    runtime_config = dict(config)
    runtime_config["workers"] = int(os.environ.get("VAL_EVAL_DATALOADER_NUM_WORKERS", "2") or 2)
    runtime_config["gpu"] = int(gpu_override) if gpu_override is not None and torch.cuda.is_available() else (None if not torch.cuda.is_available() else runtime_config.get("gpu"))
    runtime_args = build_runtime_args(runtime_config, str(first_trial_dir))
    runtime_args.workers = int(runtime_config["workers"])
    runtime_args.prefetch_factor = 2
    runtime_args.gpu = runtime_args.gpu if torch.cuda.is_available() else None
    set_global_determinism(seed)
    logger = _NullLogger()
    model, _ = build_model(runtime_args, logger)  # type: ignore[arg-type]
    train_loader, val_loader, _, _, _, _ = build_dataloaders(runtime_args)
    del train_loader
    _enable_phase2b_image_cache(val_loader.dataset, label=f"{dataset.upper()}_checkpoint_val")
    view_transform = build_transform_from_view_spec(view, seed=seed)
    view_dataset = clone_dataset_with_transform(val_loader.dataset, view_transform)
    loader_kwargs = {
        "batch_size": int(runtime_args.batch_size),
        "shuffle": False,
        "num_workers": int(runtime_args.workers),
        "pin_memory": runtime_args.gpu is not None,
    }
    if int(runtime_args.workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(runtime_args.prefetch_factor)
    view_loader = DataLoader(view_dataset, **loader_kwargs)
    if runtime_args.gpu is not None:
        model = model.cuda(runtime_args.gpu)

    rows: list[Dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoint_paths, key=lambda path: path.name):
        _phase2b_log(
            f"checkpoint group single-view start dataset={dataset.upper()} split={split_name} seed={seed} checkpoint={checkpoint_path} view={view.get('name') or view.get('signature') or 'view'}"
        )
        checkpoint_payload = load_checkpoint_bundle(checkpoint_path)
        epoch = int(checkpoint_payload.get("epoch", 0))
        model.load_state_dict(checkpoint_payload["state_dict"])
        score, _ = evaluate_model(view_loader, model, runtime_args, logger, str(view.get("name") or view.get("signature") or "view"), epoch)
        _phase2b_log(
            f"checkpoint group single-view done dataset={dataset.upper()} split={split_name} seed={seed} epoch={epoch} checkpoint={checkpoint_path} view={view.get('name') or view.get('signature') or 'view'}"
        )
        rows.append(
            {
                "dataset": dataset.upper(),
                "split": split_name,
                "seed": seed,
                "epoch": epoch,
                "checkpoint_id": checkpoint_path.stem,
                "view_name": view.get("name"),
                "view_signature": view.get("signature"),
                "val_score": round(float(score), 6),
                "status": "ok",
                "checkpoint_path": str(checkpoint_path),
            }
        )
    return rows


def evaluate_checkpoint_group_views_scores(
    checkpoint_paths: Sequence[Path],
    *,
    dataset: str,
    views: Sequence[Mapping[str, Any]],
    gpu_override: int | None = None,
) -> list[Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from training.inner_loop import build_dataloaders, build_model, build_runtime_args, evaluate_model, set_global_determinism
    from training.validation_protocols import clone_dataset_with_transform

    checkpoint_paths = [Path(path) for path in checkpoint_paths]
    view_specs = [dict(view) for view in views]
    if not checkpoint_paths or not view_specs:
        return []
    first_trial_dir = _trial_dir_from_checkpoint(checkpoint_paths[0])
    config = read_json(first_trial_dir / "config.json")
    seed = int(config.get("seed", 0))
    split_dir = str(config.get("split_dir") or "")
    split_name = Path(split_dir).name or "unknown_split"
    runtime_config = dict(config)
    runtime_config["workers"] = int(os.environ.get("VAL_EVAL_DATALOADER_NUM_WORKERS", "2") or 2)
    runtime_config["gpu"] = int(gpu_override) if gpu_override is not None and torch.cuda.is_available() else (None if not torch.cuda.is_available() else runtime_config.get("gpu"))
    runtime_args = build_runtime_args(runtime_config, str(first_trial_dir))
    runtime_args.workers = int(runtime_config["workers"])
    runtime_args.prefetch_factor = 2
    runtime_args.gpu = runtime_args.gpu if torch.cuda.is_available() else None
    set_global_determinism(seed)
    logger = _NullLogger()
    model, _ = build_model(runtime_args, logger)  # type: ignore[arg-type]
    train_loader, val_loader, _, _, _, _ = build_dataloaders(runtime_args)
    del train_loader
    _enable_phase2b_image_cache(val_loader.dataset, label=f"{dataset.upper()}_checkpoint_val")
    if runtime_args.gpu is not None:
        model = model.cuda(runtime_args.gpu)

    rows: list[Dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoint_paths, key=lambda path: path.name):
        _phase2b_log(
            f"checkpoint group multi-view start dataset={dataset.upper()} split={split_name} seed={seed} checkpoint={checkpoint_path} views={len(view_specs)}"
        )
        checkpoint_payload = load_checkpoint_bundle(checkpoint_path)
        epoch = int(checkpoint_payload.get("epoch", 0))
        model.load_state_dict(checkpoint_payload["state_dict"])
        for offset, view in enumerate(view_specs, start=1):
            set_global_determinism(seed + epoch + offset)
            view_transform = build_transform_from_view_spec(view, seed=seed + epoch + offset)
            view_dataset = clone_dataset_with_transform(val_loader.dataset, view_transform)
            view_loader = DataLoader(
                view_dataset,
                batch_size=int(runtime_args.batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=runtime_args.gpu is not None,
            )
            score, _ = evaluate_model(view_loader, model, runtime_args, logger, str(view.get("name") or view.get("signature") or "view"), epoch)
            rows.append(
                {
                    "dataset": dataset.upper(),
                    "split": split_name,
                    "seed": seed,
                    "epoch": epoch,
                    "checkpoint_id": checkpoint_path.stem,
                    "view_name": view.get("name"),
                    "view_signature": view.get("signature"),
                    "val_score": round(float(score), 6),
                    "status": "ok",
                    "checkpoint_path": str(checkpoint_path),
                }
            )
        _phase2b_log(
            f"checkpoint group multi-view done dataset={dataset.upper()} split={split_name} seed={seed} checkpoint={checkpoint_path} views={len(view_specs)}"
        )
    return rows


def load_checkpoint_table(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(__import__("json").loads(line))
    return rows
