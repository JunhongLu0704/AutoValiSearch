from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.search_agent import SearchAgent
from llm.backend_config import load_backend_choice
from phase1_search.aggregate_harness import run_aggregate_trial
from phase1_search.botorch_search import propose_botorch
from phase1_search.checkpoint_cleanup import copy_trial_artifacts, delete_trial_artifacts, summarize_cleanup
from phase1_search.llm_search import propose_llm
from phase1_search.random_search import propose_random
from phase1_search.search_memory import summarize_search_memory
from phase1_search.tpe_search import propose_tpe
from phase1_search.search_space import SHARED_ANCHOR_CONFIG, make_shared_anchor_proposal, repair_phase1_config
from utils.io import append_jsonl, write_csv, write_json, write_jsonl, write_text


def _default_split_dirs(repo_root: Path, dataset: str) -> list[tuple[str, str]]:
    split_root = repo_root / "splits"
    if dataset.upper() == "PACS":
        names = [
            "split_compositional_dominant_art_painting_target_sketch",
            "split_compositional_dominant_cartoon_target_art_painting",
            "split_compositional_dominant_photo_target_cartoon",
            "split_compositional_dominant_sketch_target_photo",
        ]
    else:
        names = [
            "split_compositional_dominant_caltech_target_sun",
            "split_compositional_dominant_labelme_target_caltech",
            "split_compositional_dominant_pascal_target_labelme",
            "split_compositional_dominant_sun_target_pascal",
        ]
    return [(name, str(split_root / name)) for name in names]


def _history_key(proposal: Mapping[str, Any]) -> tuple[Any, ...]:
    config = proposal.get("config") or proposal
    if not isinstance(config, Mapping):
        return ("invalid",)
    return (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))


def _validated_proposal(proposal: Mapping[str, Any], *, method: str, trial_index: int, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    config = repair_phase1_config(dict(proposal.get("config") or proposal))
    required_keys = {"lr", "lambdap", "epochp", "num_f"}
    if set(config.keys()) != required_keys:
        raise ValueError(f"{method} proposal missing required search keys")
    proposal = dict(proposal)
    proposal["proposal_id"] = str(proposal.get("proposal_id") or f"{method}_{trial_index:04d}")
    proposal["config"] = config
    proposal["proposal_source"] = str(proposal.get("proposal_source") or method.replace("_broad", ""))
    proposal["relation_to_memory"] = str(proposal.get("relation_to_memory") or "iterative refinement from memory summary")
    proposal["risk_note"] = str(proposal.get("risk_note") or "validated against search space and history")
    if _history_key(proposal) in {_history_key(item.get("proposal") or item) for item in history if isinstance(item, Mapping)}:
        proposal["risk_note"] = "duplicate proposal detected; keeping only because deterministic fallback returned a repeated region"
    return proposal


def _next_proposal(
    method: str,
    *,
    trial_index: int,
    history: Sequence[Mapping[str, Any]],
    backend=None,
    seed: int = 0,
    trace_dir: Path | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    if method == "random_broad":
        proposals = propose_random(count=1, seed=seed + trial_index, history=history)
        if not proposals:
            raise ValueError("random_broad failed to produce a proposal")
        return proposals[0]
    if method == "botorch_broad":
        proposals = propose_botorch(count=1, history=history, seed=seed + trial_index)
        if not proposals:
            raise ValueError("botorch_broad failed to produce a proposal")
        return proposals[0]
    if method == "tpe_broad":
        proposals = propose_tpe(count=1, history=history)
        if not proposals:
            raise ValueError("tpe_broad failed to produce a proposal")
        return proposals[0]
    agent = SearchAgent(backend=backend, strategy="llm", seed=seed, trace_dir=trace_dir, method=method, dataset=dataset, trial_index=trial_index)
    if backend is not None or method == "llm_broad":
        proposal = agent.propose_one(history=history)
        if backend is None and method == "llm_broad":
            proposal["proposal_source"] = "fallback"
        return proposal
    proposals = propose_llm(count=1, history=history)
    if not proposals:
        raise ValueError("llm fallback failed to produce a proposal")
    return proposals[0]


def _anchor_config_from_env() -> dict[str, Any]:
    return {
        "lr": float(os.environ.get("ANCHOR_LR", SHARED_ANCHOR_CONFIG["lr"])),
        "lambdap": float(os.environ.get("ANCHOR_LAMBDAP", SHARED_ANCHOR_CONFIG["lambdap"])),
        "epochp": int(os.environ.get("ANCHOR_EPOCHP", SHARED_ANCHOR_CONFIG["epochp"])),
        "num_f": int(os.environ.get("ANCHOR_NUM_F", SHARED_ANCHOR_CONFIG["num_f"])),
    }


def _proposal_trace_row(
    *,
    trial_index: int,
    dataset: str,
    method: str,
    proposal: Mapping[str, Any],
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "trial_index": trial_index,
        "method": method,
        "dataset": dataset,
        "proposal_source": proposal.get("proposal_source", method.replace("_broad", "")),
        "llm_called": method == "llm_broad" and proposal.get("proposal_source") != "shared_anchor",
        "llm_request_ok": None if method != "llm_broad" else proposal.get("proposal_source") in {"llm", "repair"},
        "llm_finish_reason": None,
        "llm_response_truncated": False,
        "initial_json_valid": None,
        "repair_attempts": 0,
        "final_json_valid": True,
        "candidate_count": None,
        "valid_candidate_count": None,
        "accepted_candidate_index": None,
        "fallback_used": bool(fallback_used or proposal.get("proposal_source") == "fallback"),
        "fallback_reason": None,
        "config": dict(proposal.get("config") or {}),
    }


def _best_trial(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("mean_test_acc") is not None]
    if not ok:
        return None
    ok.sort(key=lambda row: float(row["mean_test_acc"]), reverse=True)
    return ok[0]


def _cleanup_non_best_trials(
    *,
    method_out: Path,
    per_trial_rows: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
) -> tuple[int, int]:
    deleted_trials = 0
    deleted_checkpoints = 0
    for row in per_trial_rows:
        if int(row["trial_index"]) == int(best["trial_index"]):
            continue
        trial_path = method_out / "trials" / f"trial_{int(row['trial_index']):04d}"
        if trial_path.exists():
            deleted_checkpoints += len(list(trial_path.rglob("*.pth")))
            delete_trial_artifacts(trial_path)
            deleted_trials += 1
    return deleted_trials, deleted_checkpoints


def run_formal_suite(
    *,
    repo_root: Path,
    output_dir: Path,
    datasets: Sequence[str],
    methods: Sequence[str],
    count: int,
    seeds: Sequence[int],
    gpus: Sequence[int | None],
    python_executable: str,
    data_root: Path,
    backend_config: str | None = None,
    backend_name: str | None = None,
    max_workers: int = 2,
    run_type: str = "formal",
    use_shared_anchor: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = load_backend_choice(backend_config, backend_name) if backend_config else None
    if backend is None and any(str(method).lower() == "llm_broad" for method in methods):
        raise ValueError(
            "llm_broad requires a backend config. Use scripts/run_full_flow_local_llm.sh, "
            "scripts/run_server_full_experiment.sh, or pass --backend-config/--backend-name."
        )

    summary_rows: list[dict[str, Any]] = []
    curve_rows: dict[str, list[dict[str, Any]]] = {}
    backend_trace_rows: list[dict[str, Any]] = []
    suite_summary: dict[str, Any] = {"datasets": {}, "backend": backend.public_dict() if backend else None}

    for dataset in datasets:
        dataset = str(dataset).upper()
        split_dirs = _default_split_dirs(repo_root, dataset)
        image_root = str(data_root / dataset)
        dataset_method_rows: list[dict[str, Any]] = []

        for method in methods:
            method_out = output_dir / "phase1" / dataset.lower() / method
            method_out.mkdir(parents=True, exist_ok=True)
            history: list[dict[str, Any]] = []
            per_trial_rows: list[dict[str, Any]] = []
            method_backend_trace_rows: list[dict[str, Any]] = []
            search_memory = summarize_search_memory(history, budget=count)
            anchor_config = _anchor_config_from_env()

            for trial_index in range(1, count + 1):
                if use_shared_anchor and trial_index == 1:
                    proposal = make_shared_anchor_proposal(method)
                    proposal["config"] = anchor_config
                else:
                    proposal = _next_proposal(
                        method,
                        trial_index=trial_index,
                        history=history,
                        backend=backend,
                        seed=0,
                        trace_dir=method_out,
                        dataset=dataset,
                    )
                validated = _validated_proposal(proposal, method=method, trial_index=trial_index, history=history)
                write_json(method_out / f"proposal_{trial_index:04d}.json", validated)
                if method == "llm_broad" and validated.get("proposal_source") == "shared_anchor":
                    append_jsonl(method_out / "llm_proposal_trace.jsonl", _proposal_trace_row(trial_index=trial_index, dataset=dataset, method=method, proposal=validated))
                elif method == "llm_broad" and backend is None:
                    append_jsonl(method_out / "llm_proposal_trace.jsonl", _proposal_trace_row(trial_index=trial_index, dataset=dataset, method=method, proposal=validated, fallback_used=True))
                    append_jsonl(method_out / "llm_fallback_usage.jsonl", {"trial_index": trial_index, "fallback_used": True, "fallback_reason": "no_backend"})

                trial_dir = method_out / "trials" / f"trial_{trial_index:04d}"
                bn_mode = str(os.environ.get("BN_MODE", "train") or "train")
                base_config = {
                    "dataset": dataset,
                    "budget": "medium",
                    "bn_mode": bn_mode,
                    "amp": True,
                    "workers": 4,
                    "prefetch_factor": 2,
                    "bs": 128,
                    "disturb_mode": "rsw",
                    "validator_protocol": "vs",
                    "weight_decay": 1e-4,
                    "proposal_id": validated["proposal_id"],
                }
                result = run_aggregate_trial(
                    python_executable=python_executable,
                    dataset=dataset,
                    method=method,
                    proposal=validated,
                    base_config=base_config,
                    split_dirs=split_dirs,
                    image_root=image_root,
                    output_dir=trial_dir,
                    seeds=seeds,
                    gpus=gpus,
                    max_workers=max_workers,
                    run_type=run_type,
                )
                result["trial_index"] = trial_index
                result["proposal_id"] = validated["proposal_id"]
                result["proposal"] = dict(validated)
                per_trial_rows.append(result)
                history.append(
                    {
                        "trial_index": trial_index,
                        "proposal": dict(validated["config"]),
                        "config": dict(validated["config"]),
                        "proposal_source": validated.get("proposal_source"),
                        "proposal_id": validated["proposal_id"],
                        "status": result["status"],
                        "mean_test_acc": result["mean_test_acc"],
                        "failure_count": result["failure_count"],
                        "fail_reason": ",".join(result.get("failure_reasons", []) or []),
                        "reason": "aggregate trial completed" if result["status"] == "ok" else "aggregate trial failed",
                        "risk_note": "" if result["status"] == "ok" else "aggregate trial failure",
                    }
                )
                search_memory = summarize_search_memory(history, budget=count)
                write_json(method_out / "search_memory_summary.json", search_memory)
                write_jsonl(method_out / "history.jsonl", history)
                write_text(
                    method_out / "search_agent_report.md",
                    "\n".join(
                        [
                            "# Search Agent Report",
                            "",
                            f"- Dataset: {dataset}",
                            f"- Method: {method}",
                            f"- Evaluated trials: {len(history)}",
                            f"- Current best: {search_memory.get('current_best')}",
                            f"- Recent trend: {search_memory.get('recent_trend')}",
                        ]
                    ),
                )
                method_backend_trace_rows.append(
                    {
                        "trial_index": trial_index,
                        "method": method,
                        "dataset": dataset,
                        "backend_name": backend.name if backend else "deterministic_fallback",
                        "model": backend.model if backend else "offline-heuristic",
                        "base_url_redacted": backend.base_url if backend else None,
                        "json_valid": True,
                        "fallback_used": backend is None,
                        "mean_test_acc": result.get("mean_test_acc"),
                        "status": result.get("status"),
                        "formal_eligible": result.get("formal_eligible"),
                        "incomplete_checkpoint_pool": result.get("incomplete_checkpoint_pool"),
                        "proposal_source": validated.get("proposal_source"),
                    }
                )
                curve_rows.setdefault(f"{dataset}/{method}", []).append(
                    {
                        "trial_index": trial_index,
                        "mean_test_acc": result.get("mean_test_acc"),
                        "best_so_far": max(
                            [row["mean_test_acc"] for row in per_trial_rows if row.get("mean_test_acc") is not None],
                            default=None,
                        ),
                        "status": result.get("status"),
                    }
                )

            best = _best_trial(per_trial_rows)
            failed_trials = [row for row in per_trial_rows if row.get("status") != "ok"]
            best_trial_dir = method_out / "best_trial_checkpoints"
            kept_paths: list[str] = []
            missing_epoch_children: list[str] = []

            if best is not None:
                best_trial_dir.mkdir(parents=True, exist_ok=True)
                source_trial_dir = method_out / "trials" / f"trial_{int(best['trial_index']):04d}"
                source_trial_checkpoint_count = len(list(source_trial_dir.rglob("*.pth"))) if source_trial_dir.exists() else 0
                for child_row in best.get("children", []):
                    child_trial_dir = Path(child_row["trial_dir"])
                    child_name = child_trial_dir.name
                    if child_trial_dir.exists():
                        copied = copy_trial_artifacts(child_trial_dir, best_trial_dir / child_name)
                        kept_paths.extend(copied)
                        if not (child_trial_dir / "epoch_checkpoints").exists():
                            missing_epoch_children.append(child_name)
                    else:
                        missing_epoch_children.append(child_name)
                kept_paths.extend(copy_trial_artifacts(source_trial_dir, best_trial_dir / "aggregate_trial"))
                if source_trial_dir.exists():
                    delete_trial_artifacts(source_trial_dir)
                deleted_trials, deleted_checkpoints = _cleanup_non_best_trials(
                    method_out=method_out,
                    per_trial_rows=per_trial_rows,
                    best=best,
                )
                if source_trial_checkpoint_count:
                    deleted_trials += 1
                    deleted_checkpoints += source_trial_checkpoint_count
                cleanup_summary = summarize_cleanup(
                    method=method,
                    dataset=dataset,
                    best_trial_id=str(best.get("proposal_id")),
                    kept_paths=kept_paths,
                    deleted_trial_count=deleted_trials,
                    deleted_checkpoint_count=deleted_checkpoints,
                    failed_trial_checkpoint_deleted=len(failed_trials),
                    missing_epoch_checkpoint_children=missing_epoch_children,
                )
                cleanup_summary["kept_child_count"] = len(best.get("children", []))
                cleanup_summary["kept_epoch_checkpoint_count"] = len(
                    [path for path in best_trial_dir.rglob("epoch_*.pth") if "aggregate_trial" not in path.parts]
                )
                cleanup_summary["best_trial_source_deleted"] = True
                cleanup_summary["best_trial_source_deleted_checkpoint_count"] = source_trial_checkpoint_count
                cleanup_summary["formal_eligible"] = bool(best.get("formal_eligible"))
                cleanup_summary["incomplete_checkpoint_pool"] = not bool(best.get("formal_eligible"))
                write_json(method_out / "best_trial.json", best)
            else:
                cleanup_summary = summarize_cleanup(
                    method=method,
                    dataset=dataset,
                    best_trial_id=None,
                    kept_paths=[],
                    deleted_trial_count=len(per_trial_rows),
                    deleted_checkpoint_count=0,
                    failed_trial_checkpoint_deleted=len(failed_trials),
                    missing_epoch_checkpoint_children=[],
                )
                cleanup_summary["formal_eligible"] = False
                cleanup_summary["incomplete_checkpoint_pool"] = True

            write_json(method_out / "cleanup_summary.json", cleanup_summary)
            write_json(method_out / "checkpoint_cleanup_summary.json", cleanup_summary)
            write_jsonl(method_out / "failed_trials.jsonl", failed_trials)
            write_json(
                method_out / "phase1_summary.json",
                {
                    "dataset": dataset,
                    "method": method,
                    "best_trial": best,
                    "cleanup_summary": cleanup_summary,
                    "backend": backend.public_dict() if backend else None,
                    "run_type": str(run_type).lower(),
                    "search_space_size": 4096,
                    "anchor_enabled": bool(use_shared_anchor),
                    "anchor_config": anchor_config if use_shared_anchor else None,
                    "winner_best_mean_test_acc": float(best["mean_test_acc"]) if best and best.get("mean_test_acc") is not None else None,
                    "checkpoint_pool_complete": bool(best and best.get("formal_eligible")),
                },
            )
            write_jsonl(method_out / "phase1_backend_trace.jsonl", method_backend_trace_rows)
            backend_trace_rows.extend(method_backend_trace_rows)

            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "best_mean_test_acc": round(float(best["mean_test_acc"]), 6) if best and best.get("mean_test_acc") is not None else None,
                    "trial_count": len(per_trial_rows),
                    "failed_trial_count": len(failed_trials),
                    "best_trial_id": best.get("proposal_id") if best else None,
                    "formal_eligible": bool(best.get("formal_eligible")) if best else False,
                    "anchor_enabled": bool(use_shared_anchor),
                    "anchor_config": json.dumps(anchor_config, sort_keys=True) if use_shared_anchor else "",
                    "llm_proposal_count": sum(1 for row in history if row.get("proposal_source") == "llm"),
                    "repair_proposal_count": sum(1 for row in history if row.get("proposal_source") == "repair"),
                    "fallback_proposal_count": sum(1 for row in history if row.get("proposal_source") == "fallback"),
                    "failed_trials": len(failed_trials),
                }
            )
            dataset_method_rows.append(summary_rows[-1])
            suite_summary["datasets"].setdefault(dataset, {})[method] = summary_rows[-1]

        dataset_winner = max(
            (row for row in dataset_method_rows if row.get("best_mean_test_acc") is not None),
            key=lambda row: float(row["best_mean_test_acc"]),
            default=None,
        )
        write_json(
            output_dir / "phase1" / dataset.lower() / "phase1_dataset_summary.json",
            {
                "dataset": dataset,
                "search_space_size": 4096,
                "winner_best_mean_test_acc": float(dataset_winner["best_mean_test_acc"]) if dataset_winner else None,
                "checkpoint_pool_complete": bool(dataset_winner.get("formal_eligible")) if dataset_winner else False,
                "methods": dataset_method_rows,
                "run_type": str(run_type).lower(),
            },
        )

    write_csv(
        output_dir / "phase1_summary_table.csv",
        summary_rows,
        [
            "dataset",
            "method",
            "best_mean_test_acc",
            "trial_count",
            "failed_trial_count",
            "best_trial_id",
            "anchor_enabled",
            "anchor_config",
            "llm_proposal_count",
            "repair_proposal_count",
            "fallback_proposal_count",
            "failed_trials",
        ],
    )
    write_json(output_dir / "phase1_best_so_far_curves.json", curve_rows)
    write_jsonl(output_dir / "phase1_backend_trace.jsonl", backend_trace_rows)
    return suite_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal Phase I suite")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default="outputs/formal")
    parser.add_argument("--datasets", nargs="+", default=["PACS", "VLCS"])
    parser.add_argument("--methods", nargs="+", default=["llm_broad", "botorch_broad", "tpe_broad"])
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gpus", nargs="*", type=int, default=[])
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / ".." / "data"))
    parser.add_argument("--backend-config")
    parser.add_argument("--backend-name")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--run-type", choices=["formal", "smoke"], default="formal")
    parser.add_argument("--use-shared-anchor", type=int, default=int(os.environ.get("USE_SHARED_ANCHOR", "1") or 1))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gpus = list(args.gpus) if args.gpus else [None for _ in args.seeds]
    summary = run_formal_suite(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        datasets=args.datasets,
        methods=args.methods,
        count=int(args.count),
        seeds=args.seeds,
        gpus=gpus,
        python_executable=args.python_executable,
        data_root=Path(args.data_root).resolve(),
        backend_config=args.backend_config,
        backend_name=args.backend_name,
        max_workers=int(args.max_workers),
        run_type=str(args.run_type),
        use_shared_anchor=bool(int(args.use_shared_anchor)),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
