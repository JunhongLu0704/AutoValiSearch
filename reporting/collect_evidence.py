from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.io import read_json, write_json
from utils.json_utils import stable_hash

from .evidence_schema import validate_evidence_pack


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_first_existing(paths: Sequence[Path]) -> Mapping[str, Any] | None:
    for path in paths:
        if path.exists():
            loaded = read_json(path)
            if isinstance(loaded, Mapping):
                return loaded
    return None


def _extract_method_metric(rows: Sequence[Mapping[str, Any]], dataset: str, method: str, field: str) -> float | None:
    matches = [row for row in rows if str(row.get("dataset", "")).upper() == dataset.upper() and str(row.get("method", "")) == method]
    if not matches:
        return None
    matches.sort(key=lambda row: float(row.get(field, 0.0) or 0.0), reverse=True)
    value = matches[0].get(field)
    return float(value) if value is not None else None


def _extract_phase2b(rows: Sequence[Mapping[str, Any]], dataset: str, protocol_name: str) -> float | None:
    dataset_lower = dataset.lower()
    exact = [
        row
        for row in rows
        if str(row.get("protocol_name", "")).strip().lower() == protocol_name.lower()
        and str(row.get("protocol_name", "")).lower().startswith(dataset_lower)
    ]
    if exact:
        exact.sort(key=lambda row: float(row.get("selected_checkpoint_test_mean", 0.0) or 0.0), reverse=True)
        value = exact[0].get("selected_checkpoint_test_mean")
        return float(value) if value is not None else None
    return None


def _llm_protocol_rows(rows: Sequence[Mapping[str, Any]], dataset: str) -> list[Mapping[str, Any]]:
    prefix = f"{dataset.lower()}_"
    return [
        row
        for row in rows
        if str(row.get("protocol_name", "")).lower().startswith(prefix)
        and "random_validator" not in str(row.get("protocol_name", "")).lower()
        and "handcrafted" not in str(row.get("protocol_name", "")).lower()
        and "vanilla_best_val" not in str(row.get("protocol_name", "")).lower()
        and "best_test_upper_bound" not in str(row.get("protocol_name", "")).lower()
        and "final_epoch" not in str(row.get("protocol_name", "")).lower()
    ]


def _random_summary_from_files(dataset_dir: Path) -> dict[str, float]:
    summary = _load_first_existing([dataset_dir / "phase2b_random_summary.json", dataset_dir / "evaluation" / "phase2b_summary.json"])
    if isinstance(summary, Mapping):
        if "random_summary" in summary and isinstance(summary["random_summary"], Mapping):
            summary = summary["random_summary"]
        if {"random_avg", "random_median", "random_best_upper_bound", "random_std"} <= set(summary.keys()):
            return {
                "random_avg": float(summary.get("random_avg", 0.0) or 0.0),
                "random_median": float(summary.get("random_median", 0.0) or 0.0),
                "random_best_upper_bound": float(summary.get("random_best_upper_bound", 0.0) or 0.0),
                "random_std": float(summary.get("random_std", 0.0) or 0.0),
            }
    return {}


def _phase2b_dataset_names(phase2b_dir: Path, summary_table: Sequence[Mapping[str, Any]]) -> list[str]:
    names = {
        path.name.upper()
        for path in phase2b_dir.iterdir()
        if path.is_dir()
        and (
            (path / "phase2b_summary_table.csv").exists()
            or (path / "controller_safe_summary.json").exists()
            or (path / "validation_scores" / "validation_scores.csv").exists()
        )
    }
    if names:
        return sorted(names)
    baseline_suffixes = (
        "_final_epoch",
        "_vanilla_best_val",
        "_best_test_upper_bound",
        "_handcrafted_photometric_mean_minus_std",
        "_handcrafted_conservative_harmonic",
    )
    return sorted(
        {
            str(row.get("protocol_name", "")).split("_", 1)[0].upper()
            for row in summary_table
            if str(row.get("protocol_name", "")).lower().endswith(baseline_suffixes)
        }
    )


def _phase2b_dataset_rows(phase2b_dir: Path, summary_table: Sequence[Mapping[str, Any]], dataset: str) -> list[dict[str, Any]]:
    dataset_table = phase2b_dir / dataset.lower() / "phase2b_summary_table.csv"
    if dataset_table.exists():
        return _read_csv(dataset_table)
    dataset_lower = dataset.lower()
    return [dict(row) for row in summary_table if str(row.get("protocol_name", "")).lower().startswith(f"{dataset_lower}_")]


def _build_phase1_metrics(phase1_dir: Path) -> dict[str, Any]:
    summary_table = _read_csv(phase1_dir / "phase1_summary_table.csv")
    if not summary_table:
        raise FileNotFoundError(str(phase1_dir / "phase1_summary_table.csv"))
    curves = read_json(phase1_dir / "phase1_best_so_far_curves.json") if (phase1_dir / "phase1_best_so_far_curves.json").exists() else {}
    metrics = {}
    datasets = sorted({str(row.get("dataset", "")).upper() for row in summary_table if str(row.get("dataset", "")).strip()})
    for dataset in datasets:
        present_methods = {str(row.get("method", "")) for row in summary_table if str(row.get("dataset", "")).upper() == dataset}
        if "llm_broad" not in present_methods:
            raise FileNotFoundError(f"phase1_summary_table missing llm_broad rows for {dataset}")
        metrics[dataset.lower()] = {
            "llm_best": float(_extract_method_metric(summary_table, dataset, "llm_broad", "best_mean_test_acc") or 0.0),
            "random_best": float(_extract_method_metric(summary_table, dataset, "random_broad", "best_mean_test_acc") or 0.0) if "random_broad" in present_methods else None,
            "tpe_best": float(_extract_method_metric(summary_table, dataset, "tpe_broad", "best_mean_test_acc") or 0.0) if "tpe_broad" in present_methods else None,
            "trial_count": max((int(row.get("trial_count", 0) or 0) for row in summary_table if str(row.get("dataset", "")).upper() == dataset), default=0),
            "available_methods": sorted(present_methods),
        }
    return {
        "task": "training configuration search",
        "search_space_size": 4096,
        "budget": 24,
        "trial_semantics": "one config x 4 splits x 2 seeds",
        "methods": ["random_broad", "tpe_broad", "llm_broad"],
        "datasets": datasets,
        "metrics": metrics,
        "artifact_paths": {
            "summary_table": str(phase1_dir / "phase1_summary_table.csv"),
            "best_so_far_curves": str(phase1_dir / "phase1_best_so_far_curves.json"),
            "backend_trace": str(phase1_dir / "phase1_backend_trace.jsonl"),
        },
        "curve_count": len(curves) if isinstance(curves, Mapping) else 0,
    }


def _build_phase2b_metrics(phase2b_dir: Path) -> dict[str, Any]:
    summary_table = _read_csv(phase2b_dir / "phase2b_summary_table.csv")
    if not summary_table:
        raise FileNotFoundError(str(phase2b_dir / "phase2b_summary_table.csv"))
    random_summary = _random_summary_from_files(phase2b_dir)
    metrics: dict[str, dict[str, float]] = {}
    datasets = _phase2b_dataset_names(phase2b_dir, summary_table)
    for dataset in datasets:
        dataset_dir = phase2b_dir / dataset.lower()
        dataset_rows = _phase2b_dataset_rows(phase2b_dir, summary_table, dataset)
        if not dataset_rows:
            continue
        dataset_summary = _load_first_existing([dataset_dir / "phase2b_summary.json"]) or {}
        controller_safe_summary = _load_first_existing([dataset_dir / "controller_safe_summary.json"]) or {}
        checkpoint_pool = dict(dataset_summary.get("checkpoint_pool", {})) if isinstance(dataset_summary, Mapping) else {}
        checkpoint_count = int(checkpoint_pool.get("actual_checkpoint_count", 0) or 0)
        if checkpoint_count <= 0:
            checkpoint_count = len(_read_csv(dataset_dir / "validation_scores" / "validation_scores.csv"))
        policy_result_rows = _read_csv(dataset_dir / "policy_results.csv")
        llm_rows = _llm_protocol_rows(dataset_rows, dataset) or policy_result_rows
        vanilla = _extract_phase2b(dataset_rows, dataset, f"{dataset.lower()}_vanilla_best_val") or 0.0
        best_test_upper_bound = _extract_phase2b(dataset_rows, dataset, f"{dataset.lower()}_best_test_upper_bound") or 0.0
        final_epoch = _extract_phase2b(dataset_rows, dataset, f"{dataset.lower()}_final_epoch") or 0.0
        handcrafted = _extract_phase2b(dataset_rows, dataset, f"{dataset.lower()}_handcrafted_photometric_mean_minus_std") or 0.0
        conservative = _extract_phase2b(dataset_rows, dataset, f"{dataset.lower()}_handcrafted_conservative_harmonic") or 0.0
        safe_row = dict(controller_safe_summary.get("controller_safe_validator", {})) if isinstance(controller_safe_summary, Mapping) else {}
        llm_designed_row = dict(controller_safe_summary.get("llm_designed_best", {})) if isinstance(controller_safe_summary, Mapping) else {}
        llm_deployable_row = dict(controller_safe_summary.get("llm_deployable_best", {})) if isinstance(controller_safe_summary, Mapping) else {}
        llm_best = float(
            llm_designed_row.get(
                "selected_checkpoint_test_mean",
                max((float(row.get("selected_checkpoint_test_mean", 0.0) or 0.0) for row in llm_rows), default=0.0),
            )
            or 0.0
        )
        llm_regret = float(
            llm_designed_row.get(
                "selection_regret",
                min((float(row.get("selection_regret", 0.0) or 0.0) for row in llm_rows), default=0.0),
            )
            or 0.0
        )
        controller_safe = float(safe_row.get("deployable_selected_test_mean", safe_row.get("selected_checkpoint_test_mean", max(vanilla, handcrafted, conservative, llm_best))) or 0.0)
        metrics[dataset.lower()] = {
            "checkpoint_count": float(checkpoint_count),
            "final_epoch": float(final_epoch),
            "vanilla_selected_test": float(vanilla),
            "best_test_upper_bound": float(best_test_upper_bound),
            "handcrafted_photometric_mean_minus_std": float(handcrafted),
            "handcrafted_conservative_harmonic": float(conservative),
            "llm_designed_best": float(llm_designed_row.get("selected_checkpoint_test_mean", llm_best) or 0.0),
            "llm_deployable_best": float(llm_deployable_row.get("deployable_selected_test_mean", llm_deployable_row.get("selected_checkpoint_test_mean", llm_best)) or 0.0),
            "controller_safe_validator": float(controller_safe),
            "checkpoint_pool_complete": bool(checkpoint_pool.get("complete", False)),
            **random_summary,
            "llm_selected_test": float(llm_best),
            "improvement_over_vanilla": round(float(llm_best - vanilla), 6),
            "upper_bound_gap": round(float(best_test_upper_bound - vanilla), 6),
            "vanilla_regret": round(float(next((float(row.get("selection_regret", 0.0) or 0.0) for row in dataset_rows if str(row.get("protocol_name", "")) == f"{dataset.lower()}_vanilla_best_val"), 0.0)), 6),
            "llm_regret": round(float(llm_regret), 6),
            "regret_reduction": round(float(next((float(row.get("selection_regret", 0.0) or 0.0) for row in dataset_rows if str(row.get("protocol_name", "")) == f"{dataset.lower()}_vanilla_best_val"), 0.0) - llm_regret), 6),
        }
    artifact_paths = {
        "summary_table": str(phase2b_dir / "phase2b_summary_table.csv"),
        "protocol_results": str(phase2b_dir / "phase2b_protocol_results.csv"),
        "policy_traces": {
            dataset.lower(): str(phase2b_dir / dataset.lower() / "policy_search_trace.jsonl")
            for dataset in datasets
            if (phase2b_dir / dataset.lower() / "policy_search_trace.jsonl").exists()
        },
        "view_caches": {
            dataset.lower(): str(phase2b_dir / dataset.lower() / "view_cache")
            for dataset in datasets
            if (phase2b_dir / dataset.lower() / "view_cache").exists()
        },
    }
    random_summary_path = phase2b_dir / "phase2b_random_summary.json"
    if random_summary_path.exists():
        artifact_paths["random_summary"] = str(random_summary_path)

    return {
        "task": "controlled validation policy design for checkpoint selection",
        "policy_interface": "JSON-only policy DSL with validation views, normalization, aggregation, epoch selection, safety fallback, diversity signatures, and cached real view execution.",
        "checkpoint_count": len(summary_table),
        "main_baselines": [
            "final_epoch",
            "vanilla_best_val",
            "best_test_upper_bound",
            "handcrafted_photometric_mean_minus_std",
            "handcrafted_conservative_harmonic",
            "llm_designed",
            "controller_safe_validator",
        ],
        "datasets": datasets,
        "metrics": metrics,
        "artifact_paths": artifact_paths,
    }


def _build_backend_comparison(backend_compare_dir: Path | None) -> dict[str, Any]:
    if backend_compare_dir is None:
        return {"available": False, "table_path": None, "summary_path": None, "table": []}
    table_path = backend_compare_dir / "backend_comparison_table.csv"
    summary_path = backend_compare_dir / "backend_comparison_summary.json"
    if not table_path.exists() and not summary_path.exists():
        return {"available": False, "table_path": str(table_path), "summary_path": str(summary_path), "table": []}
    table = _read_csv(table_path)
    summary = _load_first_existing([summary_path]) or {}
    return {
        "available": True,
        "table_path": str(table_path) if table_path.exists() else None,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "table": table,
        "summary": dict(summary),
    }


def build_evidence_pack(
    root: Path,
    *,
    mode: str = "formal",
    use_example_artifacts: bool = False,
    phase1_dir: str | None = None,
    phase2b_dir: str | None = None,
    backend_compare_dir: str | None = None,
    example_artifacts_dir: str | None = None,
) -> dict[str, Any]:
    mode = "demo" if use_example_artifacts else str(mode).lower()
    if mode == "demo":
        examples_root = Path(example_artifacts_dir or root / "examples" / "artifacts")
        example_candidates = [examples_root / "evidence_pack.example.json"]
        for example in example_candidates:
            if example.exists():
                pack = read_json(example)
                validate_evidence_pack(pack)
                if "evidence_hash" not in pack or not pack.get("evidence_hash"):
                    pack["evidence_hash"] = stable_hash(pack)
                return pack
        raise FileNotFoundError(str(example_candidates[0]))

    formal_root = root / "outputs" / "formal"
    phase1_path = Path(phase1_dir).resolve() if phase1_dir else formal_root / "phase1"
    phase2b_path = Path(phase2b_dir).resolve() if phase2b_dir else formal_root / "phase2b"
    backend_compare_path = Path(backend_compare_dir).resolve() if backend_compare_dir else None

    if not phase1_path.exists():
        raise FileNotFoundError(str(phase1_path))
    if not phase2b_path.exists():
        raise FileNotFoundError(str(phase2b_path))

    phase1 = _build_phase1_metrics(phase1_path)
    phase2b = _build_phase2b_metrics(phase2b_path)
    backend_comparison = _build_backend_comparison(backend_compare_path)

    phase1_summary = _load_first_existing(
        [
            phase1_path / "phase1" / "pacs" / "llm_broad" / "phase1_summary.json",
            phase1_path / "phase1" / "vlcs" / "llm_broad" / "phase1_summary.json",
        ]
    ) or {}
    run_type = str(phase1_summary.get("run_type", "formal")).lower()

    val_brief = _load_first_existing(
        [
            phase2b_path / "pacs" / "val_design_brief.json",
            phase2b_path / "vlcs" / "val_design_brief.json",
        ]
    ) or {}
    val_num_runs = int(val_brief.get("num_runs", 0) or 0)
    val_rounds = int(val_brief.get("rounds", 0) or 0)
    datasets_present = sorted({str(dataset).upper() for dataset in phase1.get("datasets", []) or []} | {str(dataset).upper() for dataset in phase2b.get("datasets", []) or []})
    methods_present = sorted({str(method) for method in next(iter(phase1.get("metrics", {}).values()), {}).get("available_methods", []) or []})
    checkpoint_pool_complete = {dataset.upper(): bool(metrics.get("checkpoint_pool_complete", False)) for dataset, metrics in phase2b.get("metrics", {}).items()}
    formal_performance_claims_allowed = (
        run_type == "formal"
        and datasets_present == ["PACS", "VLCS"]
        and {"random_broad", "tpe_broad", "llm_broad"}.issubset(set(methods_present or ["random_broad", "tpe_broad", "llm_broad"]))
        and max((int(row.get("trial_count", 0) or 0) for row in phase1["metrics"].values()), default=0) == 24
        and val_num_runs == 3
        and val_rounds == 12
        and all(checkpoint_pool_complete.get(dataset, False) for dataset in ["PACS", "VLCS"])
    )

    allowed_claims: list[str] = []
    if formal_performance_claims_allowed:
        phase1_all = all(
            row.get("llm_best") is not None and row.get("random_best") is not None and row.get("tpe_best") is not None and row["llm_best"] > row["random_best"] and row["llm_best"] > row["tpe_best"]
            for row in phase1["metrics"].values()
        )
        phase2b_all = all(
            row.get("llm_selected_test") is not None and row.get("vanilla_selected_test") is not None and row["llm_selected_test"] > row["vanilla_selected_test"]
            for row in phase2b["metrics"].values()
        )
        if phase1_all:
            allowed_claims.append("Phase I LLM Search Agent outperforms Random and TPE on PACS/VLCS under the fixed 24-trial budget.")
        if phase2b_all:
            allowed_claims.append("Phase II-B LLM-designed validator improves over standard vanilla best-val checkpoint selection on PACS/VLCS.")
        if not allowed_claims:
            allowed_claims.append("The formal run does not support the main performance claims.")
    else:
        allowed_claims.append("This is a small-budget smoke run. It verifies that the local pipeline can execute Phase I, Phase II-B, and Phase III end-to-end. It is not the formal PACS/VLCS performance conclusion.")

    run_metadata = {
        "run_type": run_type,
        "max_trials": max((int(row.get("trial_count", 0) or 0) for row in phase1["metrics"].values()), default=0),
        "val_num_runs": val_num_runs,
        "val_rounds": val_rounds,
        "datasets": datasets_present,
        "methods": methods_present,
        "checkpoint_pool_complete": checkpoint_pool_complete,
        "formal_performance_claims_allowed": formal_performance_claims_allowed,
        "phase2b_checkpoint_counts": {dataset.upper(): int(metrics.get("checkpoint_count", 0) or 0) for dataset, metrics in phase2b["metrics"].items()},
    }

    agent_traces = {
        "search_agent": {
            "trace_path": str(phase1_path / "phase1_backend_trace.jsonl"),
            "memory_summary_paths": [
                str(phase1_path / dataset / "llm_broad" / "search_memory_summary.json")
                for dataset in ["pacs", "vlcs"]
                if (phase1_path / dataset / "llm_broad" / "search_memory_summary.json").exists()
            ],
        },
        "val_designer_agent": {
            "pacs_trace_path": str(phase2b_path / "pacs" / "val_strategy_trace.md"),
            "vlcs_trace_path": str(phase2b_path / "vlcs" / "val_strategy_trace.md"),
        },
        "report_agent": {
            "trace_path": str(formal_root / "report_agent" / "report_agent_trace.jsonl"),
        },
    }

    pack = {
        "project": {
            "name": "AutoValiSearch",
            "version": "formal",
            "description": "Three-agent LLM system for experiment optimization and validation design.",
        },
        "architecture": {
            "agents": [
                {"name": "Search Agent", "role": "training-configuration search", "phase": "Phase I"},
                {"name": "Val Designer Agent", "role": "validation policy design for checkpoint selection", "phase": "Phase II-B"},
                {"name": "Evidence Reporter", "role": "deterministic evidence-based research reporting", "phase": "Phase III"},
            ],
            "controller_features": [
                "bounded search space",
                "validation policy DSL",
                "real validation view executor",
                "view cache",
                "policy diversity signatures",
                "per-round policy trace",
                "controller-safe validator fallback",
                "schema validation",
                "repair/retry/fallback",
                "artifact persistence",
                "memory summary",
            ],
            "backend_support": [
                "cloud OpenAI-compatible API",
                "local OpenAI-compatible LLM endpoint",
            ],
        },
        "phase1": phase1,
        "phase2b": phase2b,
        "agent_traces": agent_traces,
        "backend_comparison": backend_comparison,
        "run_metadata": run_metadata,
        "allowed_claims": allowed_claims,
        "not_allowed_claims": [
            "Do not claim full AutoSOTA.",
            "Do not claim fully autonomous scientific discovery.",
            "Do not claim automatic architecture invention.",
            "Do not claim Phase II-B beats random best-of-k as a deployable baseline.",
            "Do not claim Phase II-A is the main successful result.",
        ],
    }
    validate_evidence_pack(pack)
    pack["evidence_hash"] = stable_hash(pack)
    return pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect AutoValiSearch evidence pack")
    parser.add_argument("--root", default=".")
    parser.add_argument("--phase1-dir")
    parser.add_argument("--phase2b-dir")
    parser.add_argument("--backend-compare-dir")
    parser.add_argument("--example-artifacts-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["formal", "smoke", "demo"], default="formal")
    parser.add_argument("--use-example-artifacts", action="store_true", help="Compatibility alias for --mode demo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pack = build_evidence_pack(
        Path(args.root).resolve(),
        mode=args.mode,
        use_example_artifacts=bool(args.use_example_artifacts),
        phase1_dir=args.phase1_dir,
        phase2b_dir=args.phase2b_dir,
        backend_compare_dir=args.backend_compare_dir,
        example_artifacts_dir=args.example_artifacts_dir,
    )
    write_json(args.output, pack)
    print(json.dumps({"output": str(Path(args.output).resolve()), "evidence_hash": pack.get("evidence_hash")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
