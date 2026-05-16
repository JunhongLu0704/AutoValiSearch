from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from llm.backend_config import load_backend_choice
from phase2b_validation.baseline_protocols import build_phase2b_baseline_protocols, summarize_random_baselines
from phase2b_validation.build_checkpoint_scores import build_checkpoint_scores
from phase2b_validation.build_validation_brief import build_validation_brief
from phase2b_validation.evaluate_protocols import evaluate_protocols_against_scores
from phase2b_validation.run_val_designer import run_policy_feedback_loop
from utils.io import read_json, write_csv, write_json, write_jsonl, write_text


def _phase2b_log(message: str) -> None:
    print(f"[Phase II-B] {message}", flush=True)


def _read_score_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _checkpoint_pool_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (str(row.get("dataset", "")).upper(), str(row.get("split", "")), str(row.get("seed", "")))
        by_group[key] = by_group.get(key, 0) + 1
    expected_groups = 8
    expected_rows = expected_groups * 12
    group_counts = sorted(by_group.values())
    complete = len(rows) == expected_rows and all(count == 12 for count in group_counts) and len(by_group) == expected_groups
    return {
        "actual_checkpoint_count": len(rows),
        "expected_checkpoint_count": expected_rows,
        "actual_group_count": len(by_group),
        "expected_group_count": expected_groups,
        "complete": complete,
        "group_counts": group_counts,
    }


def _phase2b_scores_path(
    *,
    dataset: str,
    override_scores: str | None,
    phase1_root: Path,
) -> Path:
    if override_scores:
        return Path(override_scores).resolve()
    return phase1_root / "phase1" / dataset.lower() / "llm_broad" / "best_trial_checkpoints"


def _checkpoint_source_status(score_source: Path) -> dict[str, Any]:
    if not score_source.exists():
        return {"exists": False, "is_dir": False, "checkpoint_count": 0, "pth_count": 0}
    if score_source.is_file():
        return {"exists": True, "is_dir": False, "checkpoint_count": 1, "pth_count": 0}
    pth_count = sum(1 for path in score_source.rglob("*.pth") if path.is_file())
    return {"exists": True, "is_dir": True, "checkpoint_count": pth_count, "pth_count": pth_count}


def _best_named_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [row for row in rows if str(row.get("protocol_name", "")).lower() == name.lower()]
    return max(matches, key=lambda row: float(row.get("selected_checkpoint_test_mean", 0.0) or 0.0), default=None)


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=lambda row: float(row.get("selected_checkpoint_test_mean", 0.0) or 0.0), default=None)


def _deployable_value(row: Mapping[str, Any] | None) -> float:
    if not row:
        return 0.0
    return float(row.get("deployable_selected_test_mean", row.get("selected_checkpoint_test_mean", 0.0)) or 0.0)


def _best_deployable_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=_deployable_value, default=None)


def _controller_safe_summary(dataset: str, rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefix = dataset.lower()
    vanilla = _best_named_row(rows, f"{prefix}_vanilla_best_val")
    upper = _best_named_row(rows, f"{prefix}_best_test_upper_bound")
    handcrafted_rows = [row for row in rows if "handcrafted" in str(row.get("protocol_name", "")).lower()]
    handcrafted_best = _best_row(handcrafted_rows)
    llm_best = _best_row(policy_rows)
    llm_deployable_best = _best_deployable_row(policy_rows)
    safe_candidates = [row for row in [vanilla, handcrafted_best, llm_deployable_best] if row]
    controller_safe = _best_deployable_row(safe_candidates) if safe_candidates else None
    return {
        "dataset": dataset,
        "vanilla_best_val": vanilla,
        "handcrafted_best": handcrafted_best,
        "llm_designed_best": llm_best,
        "llm_deployable_best": llm_deployable_best,
        "controller_safe_validator": controller_safe,
        "best_test_upper_bound": upper,
        "interpretation": "Val Designer explores validation policies, while the Controller keeps a safe fallback to the standard best-val selector when policy feedback does not support replacing it.",
    }


def run_phase2b_formal_suite(
    *,
    repo_root: Path,
    output_dir: Path,
    datasets: list[str],
    phase1_root: Path,
    num_runs: int = 1,
    rounds: int = 24,
    protocols_per_round: int = 1,
    random_validator_count: int = 0,
    backend_config: str | None = None,
    backend_name: str | None = None,
    phase2b_scores_pacs: str | None = None,
    phase2b_scores_vlcs: str | None = None,
    run_type: str = "formal",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = load_backend_choice(backend_config, backend_name) if backend_config else None
    suite_summary: dict[str, Any] = {"datasets": {}, "backend": backend.public_dict() if backend else None}
    suite_rows: list[dict[str, Any]] = []
    suite_protocol_rows: list[dict[str, Any]] = []
    strategy_trace_parts: list[str] = []
    _phase2b_log(
        f"formal suite start datasets={','.join(str(dataset).upper() for dataset in datasets)} output_dir={output_dir} backend={'yes' if backend else 'no'} run_type={run_type}"
    )
    for dataset in datasets:
        dataset = str(dataset).upper()
        dataset_dir = output_dir / "phase2b" / dataset.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        override_scores = phase2b_scores_pacs if dataset == "PACS" else phase2b_scores_vlcs if dataset == "VLCS" else None
        score_source = _phase2b_scores_path(dataset=dataset, override_scores=override_scores, phase1_root=phase1_root)
        source_status = _checkpoint_source_status(score_source)
        _phase2b_log(
            f"dataset={dataset} checkpoint_source exists={source_status['exists']} is_dir={source_status['is_dir']} checkpoint_count={source_status['checkpoint_count']}"
        )
        if not source_status["exists"]:
            raise FileNotFoundError(
                f"Missing Phase II-B checkpoint source: {score_source}. Restore the Phase I best-trial checkpoint directory or pass --phase2b-scores."
            )
        score_dir = dataset_dir / "validation_scores"
        score_dir.mkdir(parents=True, exist_ok=True)
        _phase2b_log(f"dataset={dataset} checkpoint scoring begin source={score_source}")
        if score_source.is_dir():
            rows = build_checkpoint_scores(score_source, dataset=dataset, mode="formal", failed_output_dir=score_dir)
        else:
            rows = _read_score_rows(score_source)
        _phase2b_log(f"dataset={dataset} checkpoint scoring complete rows={len(rows)}")
        if not rows:
            raise FileNotFoundError(
                f"No checkpoint weights were found under {score_source}. Phase II-B requires restored epoch_*.pth or model_best_val.pth artifacts, or a --phase2b-scores override."
            )
        checkpoint_pool = _checkpoint_pool_status(rows)
        _phase2b_log(
            f"dataset={dataset} checkpoint pool actual={checkpoint_pool['actual_checkpoint_count']} expected={checkpoint_pool['expected_checkpoint_count']} complete={checkpoint_pool['complete']}"
        )
        if str(run_type).lower() == "formal" and not checkpoint_pool["complete"]:
            raise RuntimeError(
                f"Formal Phase II-B requires a complete checkpoint pool: {checkpoint_pool['actual_checkpoint_count']} / {checkpoint_pool['expected_checkpoint_count']}"
            )
        write_jsonl(score_dir / "validation_scores.jsonl", rows)
        write_csv(
            score_dir / "validation_scores.csv",
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
        brief = build_validation_brief(dataset=dataset, summary={"source": "formal_suite", "checkpoint_count": len(rows)}, checkpoint_count=len(rows))
        brief["num_runs"] = int(num_runs)
        brief["rounds"] = int(rounds)
        brief["protocols_per_round"] = int(protocols_per_round)
        brief["run_type"] = str(run_type).lower()
        brief["checkpoint_pool_complete"] = bool(checkpoint_pool["complete"])
        brief["checkpoint_pool_actual_count"] = int(checkpoint_pool["actual_checkpoint_count"])
        brief["checkpoint_pool_expected_count"] = int(checkpoint_pool["expected_checkpoint_count"])
        write_json(dataset_dir / "val_design_brief.json", brief)
        _phase2b_log(
            f"dataset={dataset} val designer begin rounds={brief['rounds']} runs={brief['num_runs']} protocols_per_round={brief['protocols_per_round']}"
        )
        loop = run_policy_feedback_loop(
            brief,
            backend=backend,
            score_rows=rows,
            output_dir=dataset_dir,
            view_cache_dir=dataset_dir / "view_cache",
            mode="formal",
        )
        _phase2b_log(f"dataset={dataset} val designer complete policies={len(loop['policy_rounds'])}")
        write_json(dataset_dir / "round0_diagnosis.json", loop["diagnosis"])
        write_json(dataset_dir / "phase2b_val_designer_summary.json", {"dataset": dataset, "policy_count": len(loop["policy_rounds"]), "backend": backend.public_dict() if backend else None})
        write_text(dataset_dir / "val_strategy_trace.md", loop["trace_markdown"])
        write_jsonl(dataset_dir / "phase2b_val_designer_trace.jsonl", [{"dataset": dataset, "event": "policy_loop_completed", "policy_count": len(loop["policy_rounds"])}])
        llm_policies = list(loop["policies"])
        all_protocols = build_phase2b_baseline_protocols(dataset, random_count=random_validator_count)
        protocols_path = dataset_dir / "phase2b_all_protocols.json"
        write_json(protocols_path, all_protocols)

        baseline_rows, selected_rows, random_summary = evaluate_protocols_against_scores(all_protocols, rows, dataset=dataset)
        policy_rows = []
        for row in loop["policy_results"]:
            policy_rows.append(
                {
                    **dict(row),
                    "protocol_name": str(row.get("policy_name")),
                    "mean_epoch_distance_to_oracle": row.get("mean_epoch_distance_to_oracle", ""),
                }
            )
        result_rows = baseline_rows + policy_rows
        controller_safe = _controller_safe_summary(dataset, result_rows, policy_rows)
        dataset_summary = {
            "dataset": dataset,
            "protocol_count": len(all_protocols),
            "policy_count": len(policy_rows),
            "rows": result_rows,
            "checkpoint_pool": checkpoint_pool,
            "llm_designed_best": controller_safe["llm_designed_best"],
            "controller_safe_validator": controller_safe["controller_safe_validator"],
            "best_test_upper_bound": controller_safe["best_test_upper_bound"],
        }
        if random_summary:
            dataset_summary["random_summary"] = random_summary
        write_json(dataset_dir / "phase2b_summary.json", dataset_summary)
        write_json(dataset_dir / "controller_safe_summary.json", controller_safe)
        if random_summary:
            write_json(dataset_dir / "phase2b_random_summary.json", random_summary)
        write_csv(
            dataset_dir / "phase2b_summary_table.csv",
            result_rows,
            [
                "protocol_name",
                "policy_name",
                "run_idx",
                "round_idx",
                "selected_checkpoint_test_mean",
                "deployable_selected_test_mean",
                "vanilla_best_val",
                "improvement_over_vanilla",
                "deployable_improvement_over_vanilla",
                "selection_regret",
                "top3_epoch_hit_rate",
                "top5_epoch_hit_rate",
                "gap_to_best_test_upper_bound",
                "used_safety_fallback",
                "mean_epoch_distance_to_oracle",
            ],
        )
        write_csv(
            dataset_dir / "policy_results.csv",
            policy_rows,
            [
                "policy_name",
                "run_idx",
                "round_idx",
                "selected_checkpoint_test_mean",
                "deployable_selected_test_mean",
                "vanilla_best_val",
                "improvement_over_vanilla",
                "deployable_improvement_over_vanilla",
                "selection_regret",
                "top3_epoch_hit_rate",
                "top5_epoch_hit_rate",
                "gap_to_best_test_upper_bound",
                "used_safety_fallback",
            ],
        )
        write_csv(
            dataset_dir / "phase2b_selected_epoch_table.csv",
            selected_rows,
            ["protocol_name", "dataset", "split", "seed", "selected_epoch", "selected_test_acc", "oracle_epoch", "oracle_test_acc", "hit_top3", "hit_top5"],
        )
        random_rows = [row for row in result_rows if str(row["protocol_name"]).startswith(f"{dataset.lower()}_random_validator_")]
        baseline_lookup = {row["protocol_name"]: row for row in result_rows}
        summary_dir = dataset_dir / "evaluation"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            summary_dir / "phase2b_summary.json",
            {
                "dataset": dataset,
                "rows": result_rows,
                "checkpoint_pool": checkpoint_pool,
                "llm_designed_best": controller_safe["llm_designed_best"],
                "controller_safe_validator": controller_safe["controller_safe_validator"],
                "best_test_upper_bound": controller_safe["best_test_upper_bound"],
                **({"random_summary": random_summary} if random_summary else {}),
                "baseline": {
                    "final_epoch": baseline_lookup.get(f"{dataset.lower()}_final_epoch"),
                    "vanilla_best_val": baseline_lookup.get(f"{dataset.lower()}_vanilla_best_val"),
                    "best_test_upper_bound": baseline_lookup.get(f"{dataset.lower()}_best_test_upper_bound"),
                    "handcrafted_photometric_mean_minus_std": baseline_lookup.get(f"{dataset.lower()}_handcrafted_photometric_mean_minus_std"),
                    "handcrafted_conservative_harmonic": baseline_lookup.get(f"{dataset.lower()}_handcrafted_conservative_harmonic"),
                },
                "random_count": len(random_rows),
            },
        )
        write_csv(
            summary_dir / "phase2b_protocol_results.csv",
            result_rows,
            ["protocol_name", "selected_checkpoint_test_mean", "selection_regret", "top3_epoch_hit_rate", "top5_epoch_hit_rate", "mean_epoch_distance_to_oracle"],
        )
        trace_path = dataset_dir / "val_strategy_trace.md"
        if trace_path.exists():
            strategy_trace_parts.append(trace_path.read_text(encoding="utf-8"))
        suite_summary["datasets"][dataset] = {
            "checkpoint_count": len(rows),
            "protocol_count": len(all_protocols),
            "policy_count": len(policy_rows),
            "best_trial_dir": str(score_source),
            "checkpoint_pool": checkpoint_pool,
            "llm_designed_best": controller_safe["llm_designed_best"],
            "controller_safe_validator": controller_safe["controller_safe_validator"],
            "best_test_upper_bound": controller_safe["best_test_upper_bound"],
            "best_llm_protocol": controller_safe["llm_designed_best"],
            **({"random_summary": random_summary} if random_summary else {}),
        }
        _phase2b_log(f"dataset={dataset} evaluation outputs written policy_count={len(policy_rows)} protocol_count={len(all_protocols)}")
        suite_rows.extend(result_rows)
        suite_protocol_rows.extend(selected_rows)

    phase2b_root = output_dir / "phase2b"
    phase2b_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        phase2b_root / "phase2b_summary_table.csv",
        suite_rows,
        [
            "protocol_name",
            "policy_name",
            "run_idx",
            "round_idx",
            "selected_checkpoint_test_mean",
            "deployable_selected_test_mean",
            "vanilla_best_val",
            "improvement_over_vanilla",
            "deployable_improvement_over_vanilla",
            "selection_regret",
            "top3_epoch_hit_rate",
            "top5_epoch_hit_rate",
            "gap_to_best_test_upper_bound",
            "used_safety_fallback",
            "mean_epoch_distance_to_oracle",
        ],
    )
    write_csv(
        phase2b_root / "phase2b_protocol_results.csv",
        suite_protocol_rows,
        ["protocol_name", "dataset", "split", "seed", "selected_epoch", "selected_test_acc", "oracle_epoch", "oracle_test_acc", "hit_top3", "hit_top5"],
    )
    write_text(phase2b_root / "phase2b_val_strategy_trace.md", "\n\n".join(strategy_trace_parts))
    write_json(phase2b_root / "phase2b_summary.json", {"rows": suite_rows, "datasets": suite_summary["datasets"], "backend": suite_summary["backend"]})
    write_json(output_dir / "phase2b_summary.json", suite_summary)
    _phase2b_log(f"formal suite complete output_dir={output_dir} datasets={len(datasets)}")
    return suite_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal Phase II-B suite")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default="outputs/formal")
    parser.add_argument("--datasets", nargs="+", default=["PACS", "VLCS"])
    parser.add_argument("--phase1-root", default="outputs/formal")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--protocols-per-round", type=int, default=1)
    parser.add_argument("--random-validator-count", type=int, default=0)
    parser.add_argument("--backend-config")
    parser.add_argument("--backend-name")
    parser.add_argument("--phase2b-scores-pacs")
    parser.add_argument("--phase2b-scores-vlcs")
    parser.add_argument("--run-type", choices=["formal", "smoke"], default="formal")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_phase2b_formal_suite(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        datasets=args.datasets,
        phase1_root=Path(args.phase1_root).resolve(),
        num_runs=args.num_runs,
        rounds=args.rounds,
        protocols_per_round=args.protocols_per_round,
        random_validator_count=args.random_validator_count,
        backend_config=args.backend_config,
        backend_name=args.backend_name,
        phase2b_scores_pacs=args.phase2b_scores_pacs,
        phase2b_scores_vlcs=args.phase2b_scores_vlcs,
        run_type=str(args.run_type),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
