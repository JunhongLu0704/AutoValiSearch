from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from phase2b_validation.augmentation_registry import view_signature
from utils.io import write_csv


CACHE_HEADERS = ["dataset", "split", "seed", "epoch", "checkpoint_id", "view_name", "view_signature", "val_score", "status", "checkpoint_path"]


def _phase2b_log(message: str) -> None:
    print(f"[Phase II-B] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cache_path(cache_root: Path, signature: str, split: str, seed: str | int) -> Path:
    return cache_root / signature / f"split_{split}_seed_{seed}.csv"


def merge_view_cache(cache_root: Path, signature: str) -> list[dict[str, Any]]:
    view_dir = cache_root / signature
    rows: list[dict[str, Any]] = []
    for path in sorted(view_dir.glob("split_*_seed_*.csv")):
        rows.extend(_read_csv(path))
    write_csv(view_dir / "merged_scores.csv", rows, CACHE_HEADERS)
    return rows


def ensure_view_scores(
    *,
    dataset: str,
    score_rows: Sequence[Mapping[str, Any]],
    view: Mapping[str, Any],
    cache_root: str | Path,
    evaluator: Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_root = Path(cache_root)
    signature = str(view.get("signature") or view_signature(str(view["operator"]), view.get("params", {})))
    name = str(view.get("name") or signature)
    groups = sorted({(str(row["split"]), str(row["seed"])) for row in score_rows if str(row.get("dataset", "")).upper() == dataset.upper()})
    cache_hit = all(_cache_path(cache_root, signature, split, seed).exists() for split, seed in groups) and bool(groups)
    start = time.time()
    _phase2b_log(
        f"view cache single-view start dataset={dataset.upper()} view={name} signature={signature} groups={len(groups)} cache_hit={cache_hit}"
    )
    if not cache_hit:
        missing_groups = [(split, seed) for split, seed in groups if not _cache_path(cache_root, signature, split, seed).exists()]

        def run_group(split: str, seed: str) -> tuple[str, str, list[dict[str, Any]]]:
            group_rows = [row for row in score_rows if str(row.get("dataset", "")).upper() == dataset.upper() and str(row.get("split")) == split and str(row.get("seed")) == seed]
            task_rows = evaluator(dataset, group_rows, view)
            for row in task_rows:
                row["view_name"] = name
                row["view_signature"] = signature
            return split, seed, task_rows

        parallel = max(1, int(os.environ.get("VAL_EVAL_PARALLEL", "1") or 1))
        if parallel > 1 and len(missing_groups) > 1:
            with ThreadPoolExecutor(max_workers=min(parallel, len(missing_groups))) as executor:
                futures = [executor.submit(run_group, split, seed) for split, seed in missing_groups]
                for future in as_completed(futures):
                    split, seed, task_rows = future.result()
                    _phase2b_log(f"view cache single-view complete dataset={dataset.upper()} view={name} signature={signature} split={split} seed={seed} rows={len(task_rows)}")
                    write_csv(_cache_path(cache_root, signature, split, seed), task_rows, CACHE_HEADERS)
        else:
            for split, seed in missing_groups:
                split, seed, task_rows = run_group(split, seed)
                _phase2b_log(f"view cache single-view complete dataset={dataset.upper()} view={name} signature={signature} split={split} seed={seed} rows={len(task_rows)}")
                write_csv(_cache_path(cache_root, signature, split, seed), task_rows, CACHE_HEADERS)
    rows = merge_view_cache(cache_root, signature)
    rows = [{**row, "view_name": name, "view_signature": signature} for row in rows]
    write_csv(cache_root / signature / "merged_scores.csv", rows, CACHE_HEADERS)
    feedback = {
        "view_name": name,
        "view_signature": signature,
        "cache_hit": cache_hit,
        "eval_time_sec": round(time.time() - start, 4),
        "row_count": len(rows),
    }
    return rows, feedback


def ensure_view_scores_many(
    *,
    dataset: str,
    score_rows: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
    cache_root: str | Path,
    evaluator: Callable[[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache_root = Path(cache_root)
    view_meta = [
        {
            "view": dict(view),
            "signature": str(view.get("signature") or view_signature(str(view["operator"]), view.get("params", {}))),
            "name": str(view.get("name") or view.get("signature") or view_signature(str(view["operator"]), view.get("params", {}))),
        }
        for view in views
    ]
    groups = sorted({(str(row["split"]), str(row["seed"])) for row in score_rows if str(row.get("dataset", "")).upper() == dataset.upper()})
    per_view_cache_hit = {
        meta["signature"]: all(_cache_path(cache_root, meta["signature"], split, seed).exists() for split, seed in groups) and bool(groups)
        for meta in view_meta
    }
    batch_cache_hit = all(per_view_cache_hit.values()) and bool(groups)
    start = time.time()
    _phase2b_log(
        f"view cache batch start dataset={dataset.upper()} views={len(view_meta)} groups={len(groups)} batch_cache_hit={batch_cache_hit}"
    )
    if not batch_cache_hit:
        missing_groups = [
            (split, seed)
            for split, seed in groups
            if any(not _cache_path(cache_root, meta["signature"], split, seed).exists() for meta in view_meta)
        ]
        _phase2b_log(
            f"view cache batch evaluating dataset={dataset.upper()} missing_groups={len(missing_groups)} parallel={max(1, int(os.environ.get('VAL_EVAL_PARALLEL', '1') or 1))}"
        )

        def run_group(split: str, seed: str) -> tuple[str, str, dict[str, list[dict[str, Any]]]]:
            group_rows = [
                row
                for row in score_rows
                if str(row.get("dataset", "")).upper() == dataset.upper()
                and str(row.get("split")) == split
                and str(row.get("seed")) == seed
            ]
            task_rows = evaluator(dataset, group_rows, [meta["view"] for meta in view_meta])
            grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            name_by_signature = {meta["signature"]: meta["name"] for meta in view_meta}
            for row in task_rows:
                signature = str(row.get("view_signature") or row.get("view_name") or "")
                if not signature:
                    signature = next((meta["signature"] for meta in view_meta if meta["name"] == str(row.get("view_name") or "")), "")
                row["view_signature"] = signature
                row["view_name"] = name_by_signature.get(signature, str(row.get("view_name") or signature))
                grouped_rows[signature].append(row)
            return split, seed, grouped_rows

        parallel = max(1, int(os.environ.get("VAL_EVAL_PARALLEL", "1") or 1))
        if parallel > 1 and len(missing_groups) > 1:
            with ThreadPoolExecutor(max_workers=min(parallel, len(missing_groups))) as executor:
                futures = [executor.submit(run_group, split, seed) for split, seed in missing_groups]
                for future in as_completed(futures):
                    split, seed, grouped_rows = future.result()
                    _phase2b_log(
                        f"view cache batch group complete dataset={dataset.upper()} split={split} seed={seed} views={len(grouped_rows)}"
                    )
                    for signature, task_rows in grouped_rows.items():
                        write_csv(_cache_path(cache_root, signature, split, seed), task_rows, CACHE_HEADERS)
        else:
            for split, seed in missing_groups:
                split, seed, grouped_rows = run_group(split, seed)
                _phase2b_log(
                    f"view cache batch group complete dataset={dataset.upper()} split={split} seed={seed} views={len(grouped_rows)}"
                )
                for signature, task_rows in grouped_rows.items():
                    write_csv(_cache_path(cache_root, signature, split, seed), task_rows, CACHE_HEADERS)
    combined_rows: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    for meta in view_meta:
        rows = merge_view_cache(cache_root, meta["signature"])
        rows = [{**row, "view_name": meta["name"], "view_signature": meta["signature"]} for row in rows]
        write_csv(cache_root / meta["signature"] / "merged_scores.csv", rows, CACHE_HEADERS)
        combined_rows.extend(rows)
        feedback.append(
            {
                "view_name": meta["name"],
                "view_signature": meta["signature"],
                "cache_hit": per_view_cache_hit[meta["signature"]],
                "eval_time_sec": round(time.time() - start, 4),
                "row_count": len(rows),
            }
        )
    _phase2b_log(
        f"view cache batch complete dataset={dataset.upper()} views={len(view_meta)} groups={len(groups)} elapsed_sec={round(time.time() - start, 4)}"
    )
    return combined_rows, feedback
