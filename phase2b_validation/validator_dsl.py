from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ALLOWED_VIEWS = {
    "source_val",
    "color_jitter_low",
    "color_jitter_medium",
    "gaussian_blur_low",
    "gaussian_blur_medium",
    "grayscale",
    "noise_low",
    "random_resized_crop_mild",
}
ALLOWED_AGGREGATIONS = {"mean", "harmonic_mean", "mean_minus_std", "weighted_mean"}
REQUIRED_SELECTION_RULE = "select_epoch_with_max_protocol_score"
FINAL_EPOCH_SELECTION_RULE = "select_final_epoch"
TEST_EPOCH_SELECTION_RULE = "select_epoch_with_max_test_score"


@dataclass(frozen=True)
class Protocol:
    protocol_name: str
    views: List[str]
    aggregation: str
    alpha: float = 0.0
    weights: List[float] | None = None
    selection_rule: str = REQUIRED_SELECTION_RULE
    design_hypothesis: str = ""
    expected_advantage: str = ""
    risk: str = ""


def validate_protocol(raw: Mapping[str, Any]) -> None:
    if "source_val" not in list(raw.get("views") or []):
        raise ValueError("Protocol must include source_val")
    for view in raw.get("views") or []:
        if view not in ALLOWED_VIEWS:
            raise ValueError(f"Illegal view: {view}")
    if str(raw.get("aggregation", "")).lower() not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"Illegal aggregation: {raw.get('aggregation')}")
    selection_rule = str(raw.get("selection_rule", REQUIRED_SELECTION_RULE))
    if selection_rule not in {REQUIRED_SELECTION_RULE, FINAL_EPOCH_SELECTION_RULE, TEST_EPOCH_SELECTION_RULE}:
        raise ValueError("Unsupported selection rule")


def compile_protocol(raw: Mapping[str, Any]) -> Dict[str, Any]:
    validate_protocol(raw)
    protocol = Protocol(
        protocol_name=str(raw["protocol_name"]),
        views=[str(item) for item in raw.get("views", [])],
        aggregation=str(raw.get("aggregation", "mean")).lower(),
        alpha=float(raw.get("alpha", 0.0)),
        weights=list(raw.get("weights")) if raw.get("weights") is not None else None,
        selection_rule=str(raw.get("selection_rule", REQUIRED_SELECTION_RULE)),
        design_hypothesis=str(raw.get("design_hypothesis", raw.get("protocol_name", ""))),
        expected_advantage=str(raw.get("expected_advantage", "")),
        risk=str(raw.get("risk", "")),
    )
    return protocol.__dict__.copy()
