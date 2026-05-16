from __future__ import annotations

from typing import Any, Mapping, Sequence


def render_val_strategy_trace(
    diagnosis: Mapping[str, Any],
    protocol_rounds: Sequence[Mapping[str, Any]],
    feedback_rounds: Sequence[Mapping[str, Any]],
    final_summary: Mapping[str, Any],
) -> str:
    feedback_by_round = {int(row.get("round", -1)): row for row in feedback_rounds}
    lines = [
        "# Val Designer Strategy Trace",
        "",
        "## Round 0: Read and diagnose",
        "",
        f"- Purpose: diagnose why source-only validation can misselect checkpoints.",
        f"- Main problem: {diagnosis.get('diagnosis', {}).get('main_validation_problem', 'source validation may not align with test generalization')}",
        "",
    ]
    for round_payload in protocol_rounds:
        round_index = int(round_payload.get("round", 0))
        title = {
            1: "Initial protocol design",
            2: "Reflection and revision",
            3: "Final refinement",
        }.get(round_index, "Protocol design")
        lines.extend([f"## Round {round_index}: {title}", ""])
        if round_payload.get("reflection"):
            reflection = round_payload["reflection"]
            lines.append(f"- Reflection summary: {reflection}")
        if round_payload.get("round_summary"):
            lines.append(f"- Purpose: {round_payload['round_summary']}")
        protocols = round_payload.get("protocols", []) or []
        lines.append("- Protocol summary: " + ", ".join(str(item.get("protocol_name")) for item in protocols))
        feedback = feedback_by_round.get(round_index)
        if feedback:
            best = feedback.get("round_result_summary", {}).get("best_protocol")
            improvement = feedback.get("round_result_summary", {}).get("best_improvement_over_vanilla")
            lines.append(f"- Feedback summary: best protocol {best}, improvement over vanilla {improvement}.")
            lines.append(f"- Best protocol so far: {feedback.get('memory_summary', {}).get('best_protocol_so_far')}")
        lines.append("")
    lines.extend(
        [
            "## Final summary",
            "",
            f"- Selected protocol: {final_summary.get('best_protocol', {}).get('protocol_name')}",
            f"- Remaining uncertainty: {final_summary.get('remaining_uncertainty', 'dataset-specific view strength reliability')}",
        ]
    )
    return "\n".join(lines)
