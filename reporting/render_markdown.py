from __future__ import annotations

from typing import Any, Mapping

from utils.tables import markdown_table


def _agent_names(agents: Any) -> list[str]:
    names: list[str] = []
    for agent in agents or []:
        if isinstance(agent, Mapping):
            name = str(agent.get("name") or agent.get("agent") or "").strip()
            if name:
                names.append(name)
        else:
            names.append(str(agent))
    return names


def _dataset_rows(metrics: Mapping[str, Any], keys: list[str], *, fallback_key: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in sorted(metrics.keys()):
        row = {"Dataset": dataset.upper()}
        payload = dict(metrics.get(dataset, {}))
        for key in keys:
            value = payload.get(key)
            row[key] = "N/A" if value is None else f"{float(value):.2f}"
        if fallback_key and fallback_key not in row:
            row[fallback_key] = row.get(fallback_key, "N/A")
        rows.append(row)
    return rows


def render_demo_report(pack: Mapping[str, Any]) -> str:
    phase1 = pack["phase1"]["metrics"]
    phase2b = pack["phase2b"]["metrics"]
    run_metadata = pack.get("run_metadata", {})
    run_type = str(run_metadata.get("run_type", "formal")).lower()
    abstract_scope = "formal" if run_type == "formal" else "small-budget smoke"
    phase1_rows = [
        {
            "Dataset": dataset.upper(),
            "Random": "N/A" if row.get("random_best") is None else f'{row["random_best"]:.2f}',
            "TPE": "N/A" if row.get("tpe_best") is None else f'{row["tpe_best"]:.2f}',
            "LLM Search Agent": f'{row["llm_best"]:.2f}',
        }
        for dataset, row in sorted(phase1.items())
    ]
    phase2b_rows = [
        {
            "Dataset": dataset.upper(),
            "Final epoch": f'{row.get("final_epoch", 0.0):.2f}',
            "Vanilla best-val": f'{row.get("vanilla_selected_test", 0.0):.2f}',
            "Best-test upper bound": f'{row.get("best_test_upper_bound", row.get("vanilla_selected_test", 0.0)):.2f}',
            "Handcrafted": f'{row.get("handcrafted_photometric_mean_minus_std", 0.0):.2f}',
            "LLM-designed": f'{row.get("llm_selected_test", 0.0):.2f}',
            "Controller-safe": f'{row.get("controller_safe_validator", row.get("vanilla_selected_test", 0.0)):.2f}',
        }
        for dataset, row in sorted(phase2b.items())
    ]
    backend_support = ", ".join(pack.get("architecture", {}).get("backend_support", [])) or "cloud/local OpenAI-compatible backends"
    agents = _agent_names(pack.get("architecture", {}).get("agents", []))
    dataset_names = ", ".join(sorted({row["Dataset"] for row in phase1_rows} | {row["Dataset"] for row in phase2b_rows})) or "N/A"
    checkpoint_pool = run_metadata.get("checkpoint_pool_complete", {})
    claim_note = "Smoke run; formal performance claims are disabled." if not run_metadata.get("formal_performance_claims_allowed", False) else "Formal performance claims are enabled because the checkpoint pools are complete."
    return "\n".join(
        [
            "# AutoValiSearch Research Report",
            "",
            "## Abstract",
            "",
            f"AutoValiSearch is a controlled system for training configuration search, validation policy design, and deterministic evidence reporting. This report summarizes the {abstract_scope} Phase I and Phase II-B results and documents the reporting pipeline used to render the final artifacts.",
            "",
            "## System overview",
            "",
            f"- Components: {', '.join(agents) if agents else 'Search Agent, Val Designer Agent, Evidence Reporter'}.",
            f"- Backend support: {backend_support}.",
            f"- Datasets in this run: {dataset_names}.",
            f"- Run type: {run_metadata.get('run_type', 'formal')}.",
            f"- Checkpoint pools complete: {checkpoint_pool if checkpoint_pool else 'N/A'}.",
            "- Controller features: schema validation, bounded DSL, diversity signatures, view cache, per-round traces, repair/retry/fallback, artifact persistence, memory summaries.",
            "- Phase II-B uses a controlled validation policy interface: the LLM may design augmentation views, normalization, aggregation, epoch rule, and safety rule; the Controller validates, caches real validation views, evaluates policies, and enforces safety fallback.",
            "",
            "## Phase I search results",
            "",
            markdown_table(["Dataset", "Random", "TPE", "LLM Search Agent"], phase1_rows or [{"Dataset": "N/A", "Random": "N/A", "TPE": "N/A", "LLM Search Agent": "N/A"}]),
            "",
            "## Phase II-B validation design results",
            "",
            markdown_table(["Dataset", "Final epoch", "Vanilla best-val", "Best-test upper bound", "Handcrafted", "LLM-designed", "Controller-safe"], phase2b_rows or [{"Dataset": "N/A", "Final epoch": "N/A", "Vanilla best-val": "N/A", "Best-test upper bound": "N/A", "Handcrafted": "N/A", "LLM-designed": "N/A", "Controller-safe": "N/A"}]),
            "",
            "## Claims",
            "",
            *[f"- {item}" for item in pack["allowed_claims"]],
            f"- {claim_note}",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in pack.get("not_allowed_claims", [])],
            "- Best-test upper bound is the analysis upper bound for checkpoint selection.",
            "- Controller-safe validator results must not be interpreted as pure LLM wins; they include a controlled fallback to vanilla best-val when policy feedback does not support replacement.",
            "- The report does not claim fully autonomous discovery or automatic architecture invention.",
            f"- Results are bounded to {dataset_names} under the current fixed budget.",
            "",
            "## Reproducibility",
            "",
            "The pipeline reconstructs evidence from structured artifacts and generates the report bundle, claim audit, and dashboard from structured artifacts only.",
        ]
    )


def render_readme_summary(pack: Mapping[str, Any]) -> str:
    phase1_metrics = pack["phase1"]["metrics"]
    run_metadata = pack.get("run_metadata", {})
    datasets = ", ".join(sorted(dataset.upper() for dataset in phase1_metrics.keys())) or "N/A"
    first_dataset = sorted(phase1_metrics.keys())[0] if phase1_metrics else None
    llm_best = phase1_metrics.get(first_dataset, {}).get("llm_best") if first_dataset else None
    return "\n".join(
        [
            "# README Summary",
            "",
            "- Components: Search Agent, Val Designer Agent, Evidence Reporter.",
            f"- Datasets in this run: {datasets}.",
            f"- Phase I best LLM Search Agent score: {llm_best:.2f}." if llm_best is not None else "- Phase I best LLM Search Agent score: N/A.",
            f"- Phase II-B datasets covered: {', '.join(sorted(dataset.upper() for dataset in pack['phase2b']['metrics'].keys())) or 'N/A'}.",
            f"- Run type: {run_metadata.get('run_type', 'formal')}.",
            "- Best-test upper bound is treated as the analysis upper bound for checkpoint selection.",
            "- Phase II-B reports both LLM-designed policies and a controller-safe validator with fallback.",
        ]
    )


def render_resume_snippet(pack: Mapping[str, Any]) -> str:
    return (
        "AutoValiSearch: built an LLM-guided system for training-configuration search, validation-policy design, and deterministic evidence reporting; "
        "formalized PACS/VLCS experiments into structured artifacts, claim audit, and static dashboard outputs."
    )


def render_ppt_outline(pack: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PPT Outline",
            "",
            "1. Problem and motivation",
            "2. Three-agent architecture",
            "3. Formal Phase I search results",
            "4. Formal Phase II-B validation design results",
            "5. Evidence pack, claim audit, and dashboard",
            "6. Limitations and reproducibility",
        ]
    )


def render_limitations(pack: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Limitations",
            "",
            "- Phase I and Phase II-B are evaluated on PACS and VLCS under the repository's fixed-budget formal suite.",
            "- The Evidence Reporter is deterministic and evidence-based; it does not perform new training or generate new checkpoints.",
            "- Best-test upper bound remains the analysis upper bound, while vanilla best-val remains a formal baseline.",
            "- Backend comparison is optional and only reported when formal artifacts are available.",
        ]
    )
