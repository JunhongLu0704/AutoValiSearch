from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _config_key(config: Mapping[str, Any]) -> tuple[Any, ...]:
    return (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))


@dataclass(frozen=True)
class SearchMemorySummary:
    num_trials: int
    current_best: dict[str, Any] | None
    top_configs: list[dict[str, Any]]
    failed_or_low_quality_regions: list[str]
    under_explored_regions: list[str]
    recent_trend: str
    remaining_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_trials": self.num_trials,
            "current_best": self.current_best,
            "top_configs": self.top_configs,
            "failed_or_low_quality_regions": self.failed_or_low_quality_regions,
            "under_explored_regions": self.under_explored_regions,
            "recent_trend": self.recent_trend,
            "remaining_budget": self.remaining_budget,
        }


def summarize_search_memory(
    trials: Sequence[Mapping[str, Any]],
    *,
    budget: int = 24,
    top_k: int = 5,
) -> dict[str, Any]:
    ok_trials = [dict(row) for row in trials if row.get("status") == "ok" and row.get("mean_test_acc") is not None]
    ok_trials.sort(key=lambda row: float(row["mean_test_acc"]), reverse=True)

    top_configs = [
        {
            "config": dict(row.get("proposal", row.get("config", {}))),
            "mean_test_acc": float(row["mean_test_acc"]),
            "trial_index": int(row.get("trial_index", row.get("round", index + 1))),
        }
        for index, row in enumerate(ok_trials[:top_k])
    ]
    current_best = top_configs[0] if top_configs else None

    failed_regions: list[str] = []
    failure_reasons: list[str] = []
    high_risk_regions: list[str] = []
    for row in trials:
        config = row.get("proposal") or row.get("config") or {}
        score = row.get("mean_test_acc")
        if row.get("status") != "ok":
            reason = str(row.get("fail_reason") or row.get("reason") or row.get("risk_note") or "failed trial")
            failure_reasons.append(reason)
            failed_regions.append(reason)
            if isinstance(config, Mapping):
                lr = float(config.get("lr", 0.0) or 0.0)
                lambdap = float(config.get("lambdap", 0.0) or 0.0)
                epochp = int(config.get("epochp", 0) or 0)
                num_f = int(config.get("num_f", 0) or 0)
                if lr >= 0.04:
                    high_risk_regions.append(f"lr={lr:g}")
                if lambdap >= 16.0 and epochp <= 2:
                    high_risk_regions.append("large lambdap with short pretraining")
                if num_f >= 8 and lr >= 0.02:
                    high_risk_regions.append("large num_f under unstable lr")
        elif isinstance(config, Mapping) and float(config.get("lr", 0.0)) >= 0.08:
            failed_regions.append("lr=0.08 is risky in this demo trace")
        elif score is not None and float(score) < 75.0:
            failed_regions.append("very small lr may underfit within the demo budget")

    seen = {_config_key(dict(row.get("proposal") or row.get("config") or {})) for row in trials}
    under_explored = []
    if not any(float(key[0] or 0.0) in {0.01, 0.02} and float(key[1] or 0.0) in {2.0, 4.0, 8.0} for key in seen):
        under_explored.append("moderate lr with lambdap 2-8")
    if not any(int(key[3] or 0) >= 5 for key in seen):
        under_explored.append("num_f >= 5 near current best")
    if not under_explored:
        under_explored.append("local refinements around the current top configurations")

    recent = [row for row in trials[-4:] if row.get("mean_test_acc") is not None]
    if len(recent) >= 2 and max(float(row["mean_test_acc"]) for row in recent) >= max(float(row["mean_test_acc"]) for row in ok_trials[:1] or recent):
        recent_trend = "best score improved in the last 4 trials"
    elif recent:
        recent_trend = "recent trials are refining around the current best"
    else:
        recent_trend = "no evaluated trials yet"

    summary = SearchMemorySummary(
        num_trials=len(trials),
        current_best=current_best,
        top_configs=top_configs,
        failed_or_low_quality_regions=list(dict.fromkeys(failed_regions))[:5],
        under_explored_regions=under_explored[:5],
        recent_trend=recent_trend,
        remaining_budget=max(int(budget) - len(trials), 0),
    )
    payload = summary.to_dict()
    payload["failure_summary"] = {
        "failed_trial_count": len([row for row in trials if row.get("status") != "ok"]),
        "main_failure_reasons": list(dict.fromkeys(failure_reasons))[:5],
        "high_risk_regions": list(dict.fromkeys(high_risk_regions or failed_regions))[:5],
    }
    payload["next_search_recommendation"] = [
        "avoid repeatedly failed high-risk regions unless explicitly justified",
        "explore stable neighborhoods around the current best config",
    ]
    return payload
