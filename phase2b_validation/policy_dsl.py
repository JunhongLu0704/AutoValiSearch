from __future__ import annotations

import json
from typing import Any, Mapping

from phase2b_validation.augmentation_registry import AugmentationError, summarize_augmentation_registry, validate_view_spec


class PolicyValidationError(ValueError):
    pass


FORBIDDEN_FIELDS = {
    "raw_test_table",
    "checkpoint_test_acc",
    "oracle_epoch",
    "epoch_test_curve",
    "direct_epoch_choice",
    "python_code",
}

NORMALIZATIONS = {
    "raw",
    "rank_within_split_seed",
    "zscore_within_split_seed",
    "minmax_within_split_seed",
    "center_by_source_val",
    "softmax_within_split_seed",
    "sigmoid_within_split_seed",
}

AGGREGATIONS = {
    "mean",
    "weighted_mean",
    "harmonic_mean",
    "geometric_mean",
    "mean_minus_std",
    "trimmed_mean",
    "softmin",
    "median",
    "max",
    "logsumexp",
}

EPOCH_SELECTIONS = {
    "argmax_score",
    "latest_within_epsilon_of_best",
    "earliest_within_epsilon_of_best",
    "smoothed_score_argmax",
}


def _scan_forbidden(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in FORBIDDEN_FIELDS:
                raise PolicyValidationError(f"forbidden field {path}.{key}")
            if str(key) == "python_code":
                raise PolicyValidationError(f"forbidden field {path}.{key}")
            _scan_forbidden(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _scan_forbidden(value, f"{path}[{index}]")
    elif isinstance(payload, str) and ("def " in payload or "import " in payload):
        raise PolicyValidationError(f"python-like code is forbidden at {path}")


def load_policy_json(payload: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PolicyValidationError(f"invalid JSON policy: {exc}") from exc
    else:
        loaded = dict(payload)
    if not isinstance(loaded, Mapping):
        raise PolicyValidationError("policy must be a JSON object")
    return validate_policy(loaded)


def validate_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    _scan_forbidden(payload)
    required = ["policy_name", "new_views", "policy", "design_hypothesis", "expected_failure_mode"]
    for key in required:
        if key not in payload:
            raise PolicyValidationError(f"missing required field: {key}")
    policy_name = str(payload.get("policy_name") or "").strip()
    if not policy_name:
        raise PolicyValidationError("policy_name is required")
    new_views_raw = payload.get("new_views")
    if not isinstance(new_views_raw, list):
        raise PolicyValidationError("new_views must be a list")
    try:
        new_views = [validate_view_spec(view) for view in new_views_raw]
    except AugmentationError as exc:
        raise PolicyValidationError(str(exc)) from exc
    new_view_names = {view["name"] for view in new_views}
    policy = dict(payload.get("policy") or {})
    for key in ["views", "normalization", "aggregation", "epoch_selection", "safety_rule"]:
        if key not in policy:
            raise PolicyValidationError(f"missing required field: policy.{key}")
    views = [str(view).strip() for view in policy.get("views") or [] if str(view).strip()]
    if "source_val" not in views:
        raise PolicyValidationError("policy.views must include source_val")
    unknown = [view for view in views if view != "source_val" and view not in new_view_names]
    if unknown:
        raise PolicyValidationError(f"policy.views references unknown views: {unknown}")
    normalization = str(policy.get("normalization"))
    if normalization not in NORMALIZATIONS:
        raise PolicyValidationError(f"unsupported normalization: {normalization}")
    aggregation = dict(policy.get("aggregation") or {})
    aggregation_type = str(aggregation.get("type") or "")
    if aggregation_type not in AGGREGATIONS:
        raise PolicyValidationError(f"unsupported aggregation: {aggregation_type}")
    if aggregation_type == "weighted_mean":
        weights = aggregation.get("weights")
        if not isinstance(weights, Mapping):
            raise PolicyValidationError("weighted_mean requires weights")
        weight_keys = {str(key) for key in weights}
        if weight_keys != set(views):
            raise PolicyValidationError("weighted_mean weights must match policy.views exactly")
        total = sum(float(value) for value in weights.values())
        if abs(total - 1.0) > 1e-6:
            raise PolicyValidationError(f"weighted_mean weights must sum to 1.0, got {total}")
        aggregation["weights"] = {str(key): float(value) for key, value in weights.items()}
    epoch_selection = dict(policy.get("epoch_selection") or {})
    epoch_type = str(epoch_selection.get("type") or "")
    if epoch_type not in EPOCH_SELECTIONS:
        raise PolicyValidationError(f"unsupported epoch_selection: {epoch_type}")
    safety_rule = dict(policy.get("safety_rule") or {})
    safety_rule["compare_with_vanilla"] = bool(safety_rule.get("compare_with_vanilla", False))
    safety_rule["fallback_to_vanilla_if_underperform"] = bool(safety_rule.get("fallback_to_vanilla_if_underperform", False))
    return {
        "policy_name": policy_name,
        "new_views": new_views,
        "policy": {
            "views": views,
            "normalization": normalization,
            "aggregation": aggregation,
            "epoch_selection": epoch_selection,
            "safety_rule": safety_rule,
        },
        "design_hypothesis": str(payload.get("design_hypothesis") or ""),
        "expected_failure_mode": str(payload.get("expected_failure_mode") or ""),
    }


def policy_schema_summary() -> dict[str, Any]:
    return {
        "required_fields": ["policy_name", "new_views", "policy.views", "policy.normalization", "policy.aggregation", "policy.epoch_selection", "policy.safety_rule", "design_hypothesis", "expected_failure_mode"],
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "allowed_augmentation_operators": summarize_augmentation_registry(),
        "normalizations": sorted(NORMALIZATIONS),
        "aggregations": sorted(AGGREGATIONS),
        "epoch_selection": sorted(EPOCH_SELECTIONS),
    }
