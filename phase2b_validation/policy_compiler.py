from __future__ import annotations

from typing import Any, Mapping
import json

from phase2b_validation.policy_dsl import validate_policy


def compile_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    compiled = validate_policy(policy)
    compiled["required_views"] = [
        {
            "name": view["name"],
            "operator": view["operator"],
            "params": dict(view["params"]),
            "signature": view["signature"],
        }
        for view in compiled["new_views"]
        if view["name"] in set(compiled["policy"]["views"])
    ]
    compiled["policy_signature"] = policy_signature(compiled)
    return compiled


def extract_required_views(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(compile_policy(policy).get("required_views", []))


def policy_signature(policy: Mapping[str, Any]) -> str:
    compiled = validate_policy(policy)
    view_signature_by_name = {view["name"]: view["signature"] for view in compiled["new_views"]}
    view_ids = ["source_val" if view == "source_val" else view_signature_by_name.get(view, view) for view in compiled["policy"]["views"]]
    aggregation = dict(compiled["policy"]["aggregation"])
    if isinstance(aggregation.get("weights"), Mapping):
        aggregation["weights"] = {
            ("source_val" if str(name) == "source_val" else view_signature_by_name.get(str(name), str(name))): value
            for name, value in dict(aggregation["weights"]).items()
        }
    payload = {
        "views": view_ids,
        "normalization": compiled["policy"]["normalization"],
        "aggregation": aggregation,
        "epoch_selection": compiled["policy"]["epoch_selection"],
        "safety_rule": compiled["policy"]["safety_rule"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
