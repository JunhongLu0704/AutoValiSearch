from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


def copy_trial_artifacts(source_trial_dir: Path, destination_dir: Path) -> list[str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    kept_paths: list[str] = []
    for rel_name in ["config.json", "result.json", "model_best_val.pth", "log.txt", "error.txt"]:
        src = source_trial_dir / rel_name
        if src.exists():
            dst = destination_dir / rel_name
            shutil.copy2(src, dst)
            kept_paths.append(str(dst))
    epoch_dir = source_trial_dir / "epoch_checkpoints"
    if epoch_dir.exists():
        dst_epoch_dir = destination_dir / "epoch_checkpoints"
        shutil.copytree(epoch_dir, dst_epoch_dir, dirs_exist_ok=True)
        kept_paths.extend(str(path) for path in dst_epoch_dir.rglob("*") if path.is_file())
    return kept_paths


def delete_trial_artifacts(trial_dir: Path) -> None:
    if trial_dir.exists():
        shutil.rmtree(trial_dir, ignore_errors=True)


def summarize_cleanup(
    *,
    method: str,
    dataset: str,
    best_trial_id: str | None,
    kept_paths: Sequence[str],
    deleted_trial_count: int,
    deleted_checkpoint_count: int,
    failed_trial_checkpoint_deleted: int,
    missing_epoch_checkpoint_children: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "dataset": dataset,
        "best_trial_id": best_trial_id,
        "kept_paths": list(kept_paths),
        "deleted_trial_count": int(deleted_trial_count),
        "deleted_checkpoint_count": int(deleted_checkpoint_count),
        "failed_trial_checkpoint_deleted": int(failed_trial_checkpoint_deleted),
        "missing_epoch_checkpoint_children": list(missing_epoch_checkpoint_children or []),
    }
