from __future__ import annotations

import random
from typing import Any, Dict, List

from utils.json_utils import stable_hash

ALLOWED_RANDOM_VIEWS = [
    "source_val",
    "color_jitter_low",
    "color_jitter_medium",
    "gaussian_blur_low",
    "gaussian_blur_medium",
    "grayscale",
    "noise_low",
    "random_resized_crop_mild",
]
ALLOWED_AGGREGATIONS = ["mean", "harmonic_mean", "mean_minus_std", "weighted_mean"]


def _random_protocol(dataset: str, index: int) -> Dict[str, Any]:
    rng = random.Random(stable_hash({"dataset": dataset, "index": index}))
    extra_views = [view for view in ALLOWED_RANDOM_VIEWS if view != "source_val"]
    rng.shuffle(extra_views)
    view_count = 2 + (index % 3)
    views = ["source_val", *extra_views[:view_count - 1]]
    aggregation = rng.choice(ALLOWED_AGGREGATIONS)
    protocol: Dict[str, Any] = {
        "protocol_name": f"{dataset.lower()}_random_validator_{index:03d}",
        "views": views,
        "aggregation": aggregation,
        "selection_rule": "select_epoch_with_max_protocol_score",
        "design_hypothesis": "randomly sampled validator for analysis only",
        "expected_advantage": "diverse perturbation coverage",
        "risk": "not tuned",
    }
    if aggregation == "mean_minus_std":
        protocol["alpha"] = round(rng.choice([0.05, 0.1, 0.15]), 3)
    elif aggregation == "weighted_mean":
        raw = [rng.uniform(0.1, 1.0) for _ in views]
        total = sum(raw) or 1.0
        protocol["weights"] = [round(weight / total, 4) for weight in raw]
    return protocol


def build_phase2b_baseline_protocols(dataset: str, *, random_count: int = 0) -> List[Dict[str, Any]]:
    dataset = str(dataset).upper()
    return [
        {
            "protocol_name": f"{dataset.lower()}_final_epoch",
            "views": ["source_val"],
            "aggregation": "mean",
            "selection_rule": "select_final_epoch",
            "design_hypothesis": "final epoch baseline",
            "expected_advantage": "simple and cheap",
            "risk": "may miss better earlier checkpoint",
        },
        {
            "protocol_name": f"{dataset.lower()}_vanilla_best_val",
            "views": ["source_val"],
            "aggregation": "mean",
            "selection_rule": "select_epoch_with_max_protocol_score",
            "design_hypothesis": "vanilla best-val baseline",
            "expected_advantage": "direct source validation",
            "risk": "can overfit source validation",
        },
        {
            "protocol_name": f"{dataset.lower()}_best_test_upper_bound",
            "views": ["source_val"],
            "aggregation": "mean",
            "selection_rule": "select_epoch_with_max_test_score",
            "design_hypothesis": "best-test upper bound",
            "expected_advantage": "oracle test selection upper bound",
            "risk": "uses test labels and is not deployable",
        },
        {
            "protocol_name": f"{dataset.lower()}_handcrafted_photometric_mean_minus_std",
            "views": ["source_val", "color_jitter_low", "color_jitter_medium", "gaussian_blur_low"],
            "aggregation": "mean_minus_std",
            "alpha": 0.1,
            "selection_rule": "select_epoch_with_max_protocol_score",
            "design_hypothesis": "moderate photometric robustness baseline",
            "expected_advantage": "stable under mild shift",
            "risk": "may be too conservative",
        },
        {
            "protocol_name": f"{dataset.lower()}_handcrafted_conservative_harmonic",
            "views": ["source_val", "gaussian_blur_low", "noise_low"],
            "aggregation": "harmonic_mean",
            "selection_rule": "select_epoch_with_max_protocol_score",
            "design_hypothesis": "conservative harmonic baseline",
            "expected_advantage": "balanced clean/robust selection",
            "risk": "can under-select later checkpoints",
        },
        *[_random_protocol(dataset, index) for index in range(int(random_count))],
    ]


def summarize_random_baselines(result_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    random_rows = [row for row in result_rows if "_random_validator_" in str(row.get("protocol_name", ""))]
    if not random_rows:
        return {}
    scores = [float(row.get("selected_checkpoint_test_mean", 0.0)) for row in random_rows]
    scores.sort()
    mid = len(scores) // 2
    median = scores[mid] if len(scores) % 2 == 1 else (scores[mid - 1] + scores[mid]) / 2.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {
        "random_avg": round(mean, 6),
        "random_median": round(median, 6),
        "random_best_upper_bound": round(max(scores), 6),
        "random_std": round(variance ** 0.5, 6),
    }
