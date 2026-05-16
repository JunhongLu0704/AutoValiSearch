from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.io import read_json, write_csv, write_json, write_jsonl

from .checkpoint_eval import VIEW_NAMES, evaluate_checkpoint_scores, resolve_phase2b_gpu_devices

DEMO_ONLY_ERROR = "Synthetic scores are not allowed in formal Phase II-B mode."


def _phase2b_log(message: str) -> None:
    print(f"[Phase II-B] {message}", flush=True)


def _checkpoint_parallel() -> int:
    raw = os.environ.get("PHASE2B_CHECKPOINT_PARALLEL", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, len([device for device in resolve_phase2b_gpu_devices() if device is not None]) or 1)


def _trial_context(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "epoch_checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _require_demo_mode(mode: str) -> None:
    if str(mode).lower() != "demo":
        raise RuntimeError(DEMO_ONLY_ERROR)


def _synthesize_view_scores(
    payload: Dict[str, Any],
    *,
    dataset: str,
    split: str,
    seed: int,
    epoch: int,
    mode: str = "formal",
) -> Dict[str, float]:
    _require_demo_mode(mode)
    base = float(payload.get("epoch_val_acc1") or payload.get("best_val_acc1") or 0.0)
    selection = float(payload.get("epoch_selection_score") or payload.get("selection_score") or base)
    delta = (epoch % 5) * 0.03 + (seed % 2) * 0.01
    return {
        "source_val": round(base, 6),
        "color_jitter_low": round(base + 0.06 + delta, 6),
        "color_jitter_medium": round(base + 0.08 + delta, 6),
        "gaussian_blur_low": round(base + 0.05 + delta / 2.0, 6),
        "gaussian_blur_medium": round(base - 0.03 + delta / 2.0, 6),
        "grayscale": round(base - 0.01 + delta / 4.0, 6),
        "noise_low": round(base + 0.02 + delta / 3.0, 6),
        "random_resized_crop_mild": round(base + 0.04 + delta / 5.0, 6),
        "selection_anchor": round(selection, 6),
    }


def _build_demo_checkpoint_scores(best_trial_dir: Path, *, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(best_trial_dir.rglob("epoch_*.pth")):
        if "aggregate_trial" in checkpoint_path.parts:
            continue
        trial_dir = _trial_context(checkpoint_path)
        result_path = trial_dir / "result.json"
        config_path = trial_dir / "config.json"
        if not result_path.exists() or not config_path.exists():
            continue
        result = read_json(result_path)
        config = read_json(config_path)
        payload = __import__("torch").load(checkpoint_path, map_location="cpu")
        split_dir = str(config.get("split_dir") or "")
        split_name = Path(split_dir).name or "unknown_split"
        seed = int(config.get("seed", 0))
        epoch = int(payload.get("epoch", 0))
        view_scores = _synthesize_view_scores(payload, dataset=dataset, split=split_name, seed=seed, epoch=epoch, mode="demo")
        row = {
            "checkpoint_id": checkpoint_path.stem,
            "dataset": dataset,
            "split": split_name,
            "seed": seed,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "status": "ok",
            "selection_anchor": round(float(view_scores["selection_anchor"]), 6),
            "test_acc": round(float(payload.get("epoch_val_acc1") or payload.get("best_val_acc1") or result.get("best_test_acc1") or 0.0), 6),
        }
        row.update({name: float(view_scores[name]) for name in VIEW_NAMES})
        rows.append(row)
    return rows


def _failed_checkpoint_row(
    checkpoint_path: Path,
    *,
    dataset: str,
    split_name: str,
    seed: int,
    epoch: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_path.stem,
        "dataset": str(dataset).upper(),
        "split": split_name,
        "seed": seed,
        "epoch": epoch,
        "checkpoint_path": str(checkpoint_path),
        "status": "fail",
        "fail_reason": error.__class__.__name__,
        "error_message": str(error),
        "selection_anchor": None,
        "source_val": None,
        "color_jitter_low": None,
        "color_jitter_medium": None,
        "gaussian_blur_low": None,
        "gaussian_blur_medium": None,
        "grayscale": None,
        "noise_low": None,
        "random_resized_crop_mild": None,
        "test_acc": None,
    }


def build_checkpoint_scores(
    best_trial_dir: Path,
    *,
    dataset: str,
    mode: str = "formal",
    failed_output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    best_trial_dir = Path(best_trial_dir)
    if str(mode).lower() == "demo":
        _phase2b_log(f"demo checkpoint scoring start dataset={str(dataset).upper()} source={best_trial_dir}")
        return _build_demo_checkpoint_scores(best_trial_dir, dataset=str(dataset).upper())

    checkpoint_paths = [
        path
        for path in sorted(best_trial_dir.rglob("epoch_*.pth"))
        if "aggregate_trial" not in path.parts
    ]
    parallel = min(_checkpoint_parallel(), len(checkpoint_paths) or 1)
    gpu_devices = resolve_phase2b_gpu_devices()
    _phase2b_log(
        f"checkpoint scoring start dataset={str(dataset).upper()} source={best_trial_dir} total_checkpoints={len(checkpoint_paths)} parallel={parallel} gpu_devices={gpu_devices}"
    )
    rows: list[dict[str, Any]] = []
    def run_checkpoint(index: int, checkpoint_path: Path) -> tuple[int, dict[str, Any]]:
        gpu_override = gpu_devices[(index - 1) % len(gpu_devices)] if gpu_devices else None
        try:
            _phase2b_log(
                f"checkpoint scoring [{index}/{len(checkpoint_paths)}] start checkpoint={checkpoint_path} gpu={gpu_override}"
            )
            row = evaluate_checkpoint_scores(checkpoint_path, dataset=str(dataset).upper(), gpu_override=gpu_override)
            _phase2b_log(
                f"checkpoint scoring [{index}/{len(checkpoint_paths)}] done checkpoint={checkpoint_path} status={row.get('status', 'ok')} epoch={row.get('epoch', 0)} gpu={gpu_override}"
            )
            return index, row
        except Exception as exc:
            trial_dir = _trial_context(checkpoint_path)
            config_path = trial_dir / "config.json"
            config = read_json(config_path) if config_path.exists() else {}
            split_dir = str(config.get("split_dir") or "")
            split_name = Path(split_dir).name or "unknown_split"
            seed = int(config.get("seed", 0))
            epoch = int(checkpoint_path.stem.split("_")[-1]) if checkpoint_path.stem.split("_")[-1].isdigit() else 0
            row = _failed_checkpoint_row(
                checkpoint_path,
                dataset=dataset,
                split_name=split_name,
                seed=seed,
                epoch=epoch,
                error=exc,
            )
            if failed_output_dir is not None:
                failed_dir = Path(failed_output_dir)
                failed_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(failed_dir / "failed_checkpoint_scores.jsonl", [row])
            _phase2b_log(f"checkpoint scoring [{index}/{len(checkpoint_paths)}] failed checkpoint={checkpoint_path} error={exc}")
            raise RuntimeError(f"Formal Phase II-B checkpoint evaluation failed for {checkpoint_path}: {exc}") from exc

    if parallel > 1 and len(checkpoint_paths) > 1:
        results_by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(run_checkpoint, index, checkpoint_path) for index, checkpoint_path in enumerate(checkpoint_paths, start=1)]
            for future in as_completed(futures):
                index, row = future.result()
                results_by_index[index] = row
        rows.extend(results_by_index[index] for index in sorted(results_by_index))
    else:
        for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            _, row = run_checkpoint(index, checkpoint_path)
            rows.append(row)
    _phase2b_log(f"checkpoint scoring complete dataset={str(dataset).upper()} rows={len(rows)}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase II-B checkpoint view scores from best-trial checkpoints")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--best-trial-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["formal", "demo"], default="formal")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    best_trial_dir = Path(args.best_trial_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        rows = build_checkpoint_scores(
            best_trial_dir,
            dataset=str(args.dataset).upper(),
            mode=str(args.mode),
            failed_output_dir=output_dir,
        )
    except Exception:
        raise
    if not rows:
        raise FileNotFoundError(
            "Missing best trial epoch checkpoints. Phase II-B requires Phase I best trial checkpoints or --phase2b-scores override."
        )
    write_csv(
        output_dir / "validation_scores.csv",
        rows,
        [
            "checkpoint_id",
            "dataset",
            "split",
            "seed",
            "epoch",
            "checkpoint_path",
            "status",
            "fail_reason",
            "error_message",
            "selection_anchor",
            "source_val",
            "color_jitter_low",
            "color_jitter_medium",
            "gaussian_blur_low",
            "gaussian_blur_medium",
            "grayscale",
            "noise_low",
            "random_resized_crop_mild",
            "test_acc",
        ],
    )
    write_json(
        output_dir / "checkpoint_index.json",
        {
            "dataset": str(args.dataset).upper(),
            "checkpoint_count": len(rows),
            "best_trial_dir": str(best_trial_dir),
            "mode": str(args.mode),
        },
    )
    write_jsonl(output_dir / "checkpoint_index.jsonl", rows)
    print(json.dumps({"dataset": str(args.dataset).upper(), "checkpoint_count": len(rows)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
