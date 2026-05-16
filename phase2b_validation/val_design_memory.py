from __future__ import annotations

from typing import Any, Mapping, Sequence


def summarize_val_design_memory(
    protocol_rounds: Sequence[Mapping[str, Any]],
    feedback_rounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocol_by_name: dict[str, Mapping[str, Any]] = {}
    for round_payload in protocol_rounds:
        for protocol in round_payload.get("protocols", []) or []:
            if isinstance(protocol, Mapping):
                protocol_by_name[str(protocol.get("protocol_name"))] = protocol

    results: list[dict[str, Any]] = []
    recent_feedback: list[dict[str, Any]] = []
    for feedback in feedback_rounds:
        for row in feedback.get("protocol_results", []) or []:
            if isinstance(row, Mapping):
                results.append(dict(row))
        recent_feedback.append(
            {
                "round": int(feedback.get("round", -1)),
                "best_protocol": str(feedback.get("round_result_summary", {}).get("best_protocol", "")),
                "best_improvement_over_vanilla": float(feedback.get("round_result_summary", {}).get("best_improvement_over_vanilla", 0.0) or 0.0),
            }
        )
    results.sort(key=lambda row: float(row.get("selected_checkpoint_test_mean", 0.0)), reverse=True)
    best = results[0] if results else None

    effective_views: set[str] = set()
    weak_views: set[str] = set()
    effective_aggs: set[str] = set()
    failed_patterns: list[str] = []
    if best:
        protocol = protocol_by_name.get(str(best.get("protocol_name")), {})
        effective_views.update(str(view) for view in protocol.get("views", []) if view != "source_val")
        effective_aggs.add(str(protocol.get("aggregation", "")))
    for row in results[-3:]:
        protocol = protocol_by_name.get(str(row.get("protocol_name")), {})
        weak_views.update(str(view) for view in protocol.get("views", []) if view not in {"source_val", ""})
        if str(protocol.get("aggregation")) == "mean_minus_std" and float(protocol.get("alpha", 0.0)) > 0.2:
            failed_patterns.append("excessive alpha in mean_minus_std can be too conservative")
        if len(protocol.get("views", []) or []) > 4:
            failed_patterns.append("too many perturbation views can dilute source validation")

    if not failed_patterns:
        failed_patterns = ["min-like or overly conservative aggregation is risky in this demo"]

    return {
        "best_protocol_so_far": best,
        "recent_feedback": recent_feedback[-4:],
        "round_count": len(protocol_rounds),
        "feedback_count": len(feedback_rounds),
        "effective_views": sorted(effective_views) or ["color_jitter_medium", "gaussian_blur_low"],
        "weak_or_risky_views": sorted(weak_views - effective_views)[:4] or ["gaussian_blur_medium"],
        "effective_aggregations": sorted(effective_aggs) or ["mean_minus_std", "harmonic_mean"],
        "failed_design_patterns": list(dict.fromkeys(failed_patterns))[:4],
        "recommended_next_changes": [
            "use source_val plus one to three moderate perturbations",
            "avoid excessive alpha",
            "compare harmonic_mean and mean_minus_std locally",
        ],
    }
