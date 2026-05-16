from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.io import read_json, write_json, write_jsonl, write_text
from utils.json_utils import stable_hash

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ChildEval:
    split_name: str
    seed: int
    gpu: int | None
    trial_dir: Path
    config_path: Path
    result_path: Path


def _mean_test_acc(result: Mapping[str, Any]) -> float | None:
    if result.get("status") != "ok":
        return None
    if result.get("best_test_acc1") is not None:
        return float(result["best_test_acc1"])
    if result.get("test_score") is not None:
        return float(result["test_score"])
    return None


def _build_child_config(
    base_config: Mapping[str, Any],
    *,
    proposal: Mapping[str, Any],
    split_dir: str,
    image_root: str,
    seed: int,
    gpu: int | None,
    trial_dir: Path,
) -> dict[str, Any]:
    config = dict(base_config)
    proposal_config = dict(proposal.get("config") or proposal)
    config.update(proposal_config)
    config.update(
        {
            "split_dir": split_dir,
            "image_root": image_root,
            "seed": int(seed),
            "trial_dir": str(trial_dir),
        }
    )
    if gpu is not None:
        config["gpu"] = int(gpu)
    return config


def _run_trial_subprocess(
    *,
    python_executable: str,
    config_path: Path,
    trial_dir: Path,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        python_executable,
        str(ROOT / "scripts" / "run_trial.py"),
        "--config",
        str(config_path),
        "--trial_dir",
        str(trial_dir),
    ]
    return subprocess.run(command, cwd=str(cwd), text=True)


def run_aggregate_trial(
    *,
    python_executable: str,
    dataset: str,
    method: str,
    proposal: Mapping[str, Any],
    base_config: Mapping[str, Any],
    split_dirs: Sequence[tuple[str, str]],
    image_root: str,
    output_dir: Path,
    seeds: Sequence[int] = (0, 1),
    gpus: Sequence[int | None] | None = None,
    max_workers: int = 8,
    run_type: str = "formal",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    child_specs: list[ChildEval] = []
    split_names = [name for name, _ in split_dirs]
    split_paths = {name: path for name, path in split_dirs}
    gpus = list(gpus or [])
    if not gpus:
        gpus = [None for _ in seeds]
    while len(gpus) < len(seeds):
        gpus.append(gpus[-1] if gpus else None)

    for seed_index, seed in enumerate(seeds):
        gpu = gpus[seed_index]
        for split_name in split_names:
            child_trial_dir = trials_dir / f"split_{split_name}_seed{seed}"
            child_trial_dir.mkdir(parents=True, exist_ok=True)
            config_path = child_trial_dir / "input_config.json"
            result_path = child_trial_dir / "result.json"
            child_specs.append(
                ChildEval(
                    split_name=split_name,
                    seed=int(seed),
                    gpu=gpu if gpu is None else int(gpu),
                    trial_dir=child_trial_dir,
                    config_path=config_path,
                    result_path=result_path,
                )
            )

    child_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    def _run_child(spec: ChildEval) -> dict[str, Any]:
        config = _build_child_config(
            base_config,
            proposal=proposal,
            split_dir=split_paths[spec.split_name],
            image_root=image_root,
            seed=spec.seed,
            gpu=spec.gpu,
            trial_dir=spec.trial_dir,
        )
        config["proposal_id"] = proposal.get("proposal_id")
        config["search_method"] = method
        config["dataset"] = dataset
        write_json(spec.config_path, config)
        completed = _run_trial_subprocess(
            python_executable=python_executable,
            config_path=spec.config_path,
            trial_dir=spec.trial_dir,
            cwd=ROOT,
        )
        result = read_json(spec.result_path) if spec.result_path.exists() else {"status": "fail", "fail_reason": "missing_result_json"}
        score = _mean_test_acc(result)
        row = {
            "split": spec.split_name,
            "seed": spec.seed,
            "gpu": spec.gpu,
            "status": result.get("status", "fail"),
            "test_acc": score,
            "trial_dir": str(spec.trial_dir),
            "config_path": str(spec.config_path),
            "result_path": str(spec.result_path),
            "returncode": completed.returncode,
            "fail_reason": result.get("fail_reason"),
            "error_message": result.get("error_message"),
            "removed_checkpoint_count": result.get("removed_checkpoint_count", 0),
            "removed_checkpoint_paths": result.get("removed_checkpoint_paths", []),
        }
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_child, spec): spec for spec in child_specs}
        for future in as_completed(future_map):
            spec = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "split": spec.split_name,
                    "seed": spec.seed,
                    "gpu": spec.gpu,
                    "status": "fail",
                    "test_acc": None,
                    "trial_dir": str(spec.trial_dir),
                    "config_path": str(spec.config_path),
                    "result_path": str(spec.result_path),
                    "returncode": None,
                    "fail_reason": "child_future_exception",
                    "error_message": str(exc),
                    "removed_checkpoint_count": 0,
                    "removed_checkpoint_paths": [],
                }
            child_rows.append(row)
            trace_rows.append(dict(row))
            write_jsonl(output_dir / "child_eval_trace.jsonl", trace_rows)

    child_rows.sort(key=lambda row: (int(row["seed"]), str(row["split"])))

    valid_rows = [row for row in child_rows if row["status"] == "ok" and row["test_acc"] is not None]
    failure_rows = [row for row in child_rows if row["status"] != "ok" or row["test_acc"] is None]
    formal_mode = str(run_type).lower() == "formal"
    expected_child_count = len(child_rows)
    valid_child_count = len(valid_rows)
    enough_valid = valid_child_count >= max(1, expected_child_count - 2)
    formal_eligible = valid_child_count == expected_child_count and not failure_rows
    mean_test_acc = sum(float(row["test_acc"]) for row in valid_rows) / len(valid_rows) if valid_rows else None
    result = {
        "trial_id": proposal.get("proposal_id") or "trial_unknown",
        "dataset": dataset,
        "method": method,
        "proposal": dict(proposal.get("config") or proposal),
        "config_hash": stable_hash(dict(proposal.get("config") or proposal)),
        "children": child_rows,
        "valid_child_count": valid_child_count,
        "expected_child_count": expected_child_count,
        "failure_count": len(failure_rows),
        "failure_reasons": sorted({str(row.get("fail_reason") or row.get("error_message") or "unknown") for row in failure_rows}),
        "formal_eligible": formal_eligible,
        "incomplete_checkpoint_pool": not formal_eligible,
        "mean_test_acc": round(mean_test_acc, 6) if mean_test_acc is not None and enough_valid and (not formal_mode or formal_eligible) else None,
        "status": "ok" if mean_test_acc is not None and enough_valid and (not formal_mode or formal_eligible) else "fail",
    }
    write_json(output_dir / "aggregate_result.json", result)
    write_jsonl(output_dir / "child_results.jsonl", child_rows)
    return result
