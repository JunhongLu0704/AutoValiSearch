from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


def executor_defaults() -> dict[str, Any]:
    return {
        "parallel": int(os.environ.get("VAL_EVAL_PARALLEL", "16") or 16),
        "gpu_devices": str(os.environ.get("VAL_EVAL_GPU_DEVICES", "0 1")).split(),
        "dataloader_num_workers": int(os.environ.get("VAL_EVAL_DATALOADER_NUM_WORKERS", "2") or 2),
    }


def demo_mapped_view_evaluator(dataset: str, group_rows: Sequence[Mapping[str, Any]], view: Mapping[str, Any]) -> list[dict[str, Any]]:
    operator = str(view.get("operator") or "identity")
    params = dict(view.get("params") or {})
    rows: list[dict[str, Any]] = []
    for row in group_rows:
        if operator == "identity":
            score = float(row.get("source_val", 0.0) or 0.0)
        elif operator == "color_jitter":
            key = "color_jitter_low" if max(float(params.get("brightness", 0.0)), float(params.get("contrast", 0.0)), float(params.get("saturation", 0.0))) <= 0.10 else "color_jitter_medium"
            score = float(row.get(key, row.get("source_val", 0.0)) or 0.0)
        elif operator == "gaussian_blur":
            key = "gaussian_blur_low" if float(params.get("sigma", 0.0)) <= 0.50 else "gaussian_blur_medium"
            score = float(row.get(key, row.get("source_val", 0.0)) or 0.0)
        elif operator == "grayscale":
            score = float(row.get("grayscale", row.get("source_val", 0.0)) or 0.0)
        elif operator == "noise":
            base = float(row.get("noise_low", row.get("source_val", 0.0)) or 0.0)
            score = base - max(0.0, float(params.get("std", 0.01)) - 0.01) * 20.0
        elif operator == "random_resized_crop":
            score = float(row.get("random_resized_crop_mild", row.get("source_val", 0.0)) or 0.0)
        elif operator == "autocontrast":
            score = float(row.get("source_val", 0.0) or 0.0) - 0.01
        elif operator == "sharpness":
            factor = float(params.get("factor", 1.0))
            score = float(row.get("source_val", 0.0) or 0.0) + (factor - 1.0) * 0.03
        elif operator == "posterize":
            bits = int(round(float(params.get("bits", 6))))
            score = float(row.get("source_val", 0.0) or 0.0) - max(0, 8 - bits) * 0.01
        elif operator == "solarize":
            threshold = float(params.get("threshold", 128.0))
            score = float(row.get("source_val", 0.0) or 0.0) - max(0.0, 192.0 - threshold) / 192.0 * 0.02
        else:
            score = float(row.get("source_val", 0.0) or 0.0)
        rows.append(
            {
                "dataset": str(row.get("dataset", dataset)).upper(),
                "split": row.get("split"),
                "seed": row.get("seed"),
                "epoch": int(row.get("epoch", 0) or 0),
                "checkpoint_id": row.get("checkpoint_id"),
                "view_name": view.get("name"),
                "view_signature": view.get("signature"),
                "val_score": round(float(score), 6),
                "status": row.get("status", "ok"),
                "checkpoint_path": row.get("checkpoint_path"),
            }
        )
    return rows


def demo_mapped_view_evaluator_many(
    dataset: str,
    group_rows: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in views:
        rows.extend(demo_mapped_view_evaluator(dataset, group_rows, view))
    return rows


def real_validation_view_evaluator(dataset: str, group_rows: Sequence[Mapping[str, Any]], view: Mapping[str, Any]) -> list[dict[str, Any]]:
    from phase2b_validation.checkpoint_eval import evaluate_checkpoint_group_view_scores

    checkpoint_paths = [Path(str(row.get("checkpoint_path") or "")) for row in sorted(group_rows, key=lambda item: int(item.get("epoch", 0) or 0))]
    return evaluate_checkpoint_group_view_scores(checkpoint_paths, dataset=dataset, view=view)


def real_validation_view_evaluator_many(
    dataset: str,
    group_rows: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from phase2b_validation.checkpoint_eval import evaluate_checkpoint_group_views_scores

    checkpoint_paths = [Path(str(row.get("checkpoint_path") or "")) for row in sorted(group_rows, key=lambda item: int(item.get("epoch", 0) or 0))]
    return evaluate_checkpoint_group_views_scores(checkpoint_paths, dataset=dataset, views=views)


def execute_view_tasks(
    *,
    dataset: str,
    score_rows: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
    evaluator=real_validation_view_evaluator,
    parallel: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    defaults = executor_defaults()
    max_workers = int(parallel or defaults["parallel"])
    groups = sorted({(str(row["split"]), str(row["seed"])) for row in score_rows if str(row.get("dataset", "")).upper() == dataset.upper()})
    tasks = [(split, seed, view) for view in views for split, seed in groups]
    output: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for split, seed, view in tasks:
            group_rows = [row for row in score_rows if str(row.get("dataset", "")).upper() == dataset.upper() and str(row.get("split")) == split and str(row.get("seed")) == seed]
            futures.append(pool.submit(evaluator, dataset, group_rows, view))
        for future in as_completed(futures):
            output.extend(future.result())
    return output, {"task_count": len(tasks), "parallel": max_workers, "epochs_per_task": sorted({len([row for row in score_rows if str(row.get('split')) == split and str(row.get('seed')) == seed]) for split, seed in groups})}
