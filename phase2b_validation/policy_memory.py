from __future__ import annotations

from typing import Any, Mapping, Sequence


def empty_policy_memory() -> dict[str, Any]:
    return {
        "best_policy_so_far": None,
        "best_deployable_policy_so_far": None,
        "policies_above_vanilla": [],
        "policies_below_vanilla": [],
        "promising_views": [],
        "harmful_views": [],
        "promising_normalizations": [],
        "failed_patterns": [],
        "recommended_next_changes": [],
    }


def update_val_policy_memory(memory: Mapping[str, Any] | None, policy: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any]:
    updated = empty_policy_memory()
    for key, value in dict(memory or {}).items():
        updated[key] = list(value) if isinstance(value, list) else value
    metrics = dict(feedback.get("policy_feedback", feedback))
    name = str(policy.get("policy_name") or metrics.get("policy_name") or "unknown_policy")
    improvement = float(metrics.get("improvement_over_vanilla", 0.0) or 0.0)
    selected = float(metrics.get("selected_checkpoint_test_mean", 0.0) or 0.0)
    deployable = float(metrics.get("deployable_selected_test_mean", selected) or selected)
    best = updated.get("best_policy_so_far")
    if not isinstance(best, Mapping) or selected > float(best.get("selected_checkpoint_test_mean", -1e9) or -1e9):
        updated["best_policy_so_far"] = {"policy_name": name, "selected_checkpoint_test_mean": selected, "improvement_over_vanilla": improvement}
    deployable_best = updated.get("best_deployable_policy_so_far")
    if not isinstance(deployable_best, Mapping) or deployable > float(deployable_best.get("deployable_selected_test_mean", -1e9) or -1e9):
        updated["best_deployable_policy_so_far"] = {"policy_name": name, "deployable_selected_test_mean": deployable, "deployable_improvement_over_vanilla": float(metrics.get("deployable_improvement_over_vanilla", improvement) or 0.0)}
    target = "policies_above_vanilla" if improvement >= 0.0 else "policies_below_vanilla"
    updated[target].append({"policy_name": name, "improvement_over_vanilla": improvement})
    views = [str(view) for view in dict(policy.get("policy", {})).get("views", []) if str(view) != "source_val"]
    view_target = "promising_views" if improvement >= 0.0 else "harmful_views"
    for view in views:
        if view not in updated[view_target]:
            updated[view_target].append(view)
    normalization = str(dict(policy.get("policy", {})).get("normalization") or "")
    if improvement >= 0.0 and normalization and normalization not in updated["promising_normalizations"]:
        updated["promising_normalizations"].append(normalization)
    if improvement < 0.0:
        pattern = f"{normalization or 'unknown normalization'} with {','.join(views) or 'source only'} underperforms vanilla"
        if pattern not in updated["failed_patterns"]:
            updated["failed_patterns"].append(pattern)
    recommendations = ["keep source_val mandatory", "use controller safety fallback", "avoid repeated policy signatures"]
    if improvement < 0.0:
        recommendations.extend(["increase source_val weight", "try rank_within_split_seed normalization"])
    updated["recommended_next_changes"] = list(dict.fromkeys(list(updated.get("recommended_next_changes") or []) + recommendations))[-8:]
    return updated


def summarize_policy_memory(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    memory = empty_policy_memory()
    for item in rounds:
        policy = item.get("policy") if isinstance(item.get("policy"), Mapping) else item
        feedback = item.get("feedback") if isinstance(item.get("feedback"), Mapping) else {}
        memory = update_val_policy_memory(memory, policy, feedback)
    return memory
