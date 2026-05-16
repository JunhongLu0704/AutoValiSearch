from __future__ import annotations

import math
from statistics import pstdev
from typing import Any, Mapping, Sequence

from .validator_dsl import compile_protocol


def _safe_float(value: Any) -> float:
    return float(value)


def harmonic_mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if float(value) > 0.0]
    if not clean:
        return 0.0
    return float(len(clean) / sum(1.0 / value for value in clean))


def score_protocol(protocol: Mapping[str, Any], view_scores: Mapping[str, float]) -> float:
    protocol = compile_protocol(protocol)
    scores = [_safe_float(view_scores[view]) for view in protocol["views"] if view in view_scores]
    if not scores:
        return 0.0
    aggregation = protocol["aggregation"]
    if aggregation == "mean":
        base = sum(scores) / len(scores)
    elif aggregation == "harmonic_mean":
        base = harmonic_mean(scores)
    elif aggregation == "mean_minus_std":
        base = (sum(scores) / len(scores)) - float(protocol["alpha"]) * (pstdev(scores) if len(scores) > 1 else 0.0)
    elif aggregation == "weighted_mean":
        weights = protocol.get("weights") or [1.0] * len(scores)
        weight_sum = sum(float(weight) for weight in weights[: len(scores)]) or 1.0
        base = sum(score * float(weight) for score, weight in zip(scores, weights)) / weight_sum
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    return round(float(base), 6)

