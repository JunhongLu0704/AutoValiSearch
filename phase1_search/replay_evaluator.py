from __future__ import annotations

from typing import Any, Mapping, Sequence


def _score_config(config: Mapping[str, Any]) -> tuple[float, str, str]:
    lr = float(config["lr"])
    lambdap = float(config["lambdap"])
    epochp = int(config["epochp"])
    num_f = int(config["num_f"])

    score = 73.8
    score += max(0.0, 1.8 - abs(lr - 0.02) * 70.0)
    score += max(0.0, 0.85 - abs(lambdap - 4.0) * 0.055)
    score += max(0.0, 0.55 - abs(epochp - 4) * 0.09)
    score += max(0.0, 0.45 - abs(num_f - 5) * 0.055)
    if lr >= 0.08:
        score -= 1.0
        risk = "lr=0.08 is risky in this replay trace"
    elif lr <= 0.00125:
        score -= 0.55
        risk = "very small lr may underfit within 12 epochs"
    else:
        risk = "moderate search risk"
    reason = "deterministic replay score from the public demo response surface"
    return round(score, 6), reason, risk


def evaluate_search_proposal(
    proposal: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    round_index: int,
) -> dict[str, Any]:
    config = dict(proposal.get("config") or proposal)
    score, reason, risk = _score_config(config)
    previous_scores = [float(row["mean_test_acc"]) for row in history if row.get("status") == "ok" and row.get("mean_test_acc") is not None]
    all_scores = sorted(previous_scores + [score], reverse=True)
    return {
        "round": int(round_index),
        "proposal": config,
        "proposal_id": proposal.get("proposal_id"),
        "hypothesis": proposal.get("hypothesis", ""),
        "status": "ok",
        "mean_test_acc": score,
        "rank_after_eval": all_scores.index(score) + 1,
        "is_new_best": not previous_scores or score > max(previous_scores),
        "reason": reason,
        "risk_note": risk,
    }
