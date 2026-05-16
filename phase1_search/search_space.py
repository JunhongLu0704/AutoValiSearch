from __future__ import annotations

import itertools
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PHASE1_SEARCH_SPACE = {
    "lr": [0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.02],
    "lambdap": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    "epochp": [0, 1, 2, 3, 4, 5, 6, 8],
    "num_f": [1, 2, 3, 4, 5, 6, 8, 10],
}

SAFE_PHASE1_PROPOSAL = {"lr": 0.005, "lambdap": 8.0, "epochp": 4, "num_f": 3}
SHARED_ANCHOR_CONFIG = {"lr": 0.01, "lambdap": 1.0, "epochp": 5, "num_f": 3}


def build_phase1_search_space() -> Dict[str, List[float | int]]:
    return {key: list(values) for key, values in PHASE1_SEARCH_SPACE.items()}


def _nearest_choice(value: Any, choices: Sequence[float | int], default: float | int) -> float | int:
    if value is None:
        return default
    template = choices[0] if choices else default
    try:
        parsed = int(float(value)) if isinstance(template, int) and not isinstance(template, bool) else float(value)
    except (TypeError, ValueError):
        return default
    if parsed in choices:
        return parsed
    return min(choices, key=lambda choice: abs(float(choice) - float(parsed)))


def repair_phase1_config(config: Dict[str, Any]) -> Dict[str, Any]:
    space = build_phase1_search_space()
    return {
        "lr": float(_nearest_choice(config.get("lr"), space["lr"], SAFE_PHASE1_PROPOSAL["lr"])),
        "lambdap": float(_nearest_choice(config.get("lambdap"), space["lambdap"], SAFE_PHASE1_PROPOSAL["lambdap"])),
        "epochp": int(_nearest_choice(config.get("epochp"), space["epochp"], SAFE_PHASE1_PROPOSAL["epochp"])),
        "num_f": int(_nearest_choice(config.get("num_f"), space["num_f"], SAFE_PHASE1_PROPOSAL["num_f"])),
    }


def validate_phase1_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    space = build_phase1_search_space()
    required = {"lr", "lambdap", "epochp", "num_f"}
    missing = required - set(config.keys())
    if missing:
        raise ValueError(f"missing config fields: {sorted(missing)}")
    out: Dict[str, Any] = {
        "lr": float(config.get("lr")),
        "lambdap": float(config.get("lambdap")),
        "epochp": int(float(config.get("epochp"))),
        "num_f": int(float(config.get("num_f"))),
    }
    for key, value in out.items():
        if value not in space[key]:
            raise ValueError(f"{key}={value} is outside search space")
    return out


def make_shared_anchor_proposal(method: str) -> Dict[str, Any]:
    return {
        "proposal_source": "shared_anchor",
        "proposal_id": f"{method}_anchor",
        "config": dict(SHARED_ANCHOR_CONFIG),
        "hypothesis": "Shared human-prior anchor used as the common initial observation.",
        "design_hypothesis": "Shared human-prior anchor used as the common initial observation.",
        "relation_to_memory": "This anchor initializes all search methods with the same first observation.",
        "risk_note": "Reasonable but intentionally imperfect initial configuration.",
    }


def grid(space: Dict[str, Sequence[float | int]] | None = None) -> List[Dict[str, Any]]:
    space = space or PHASE1_SEARCH_SPACE
    rows = []
    for lr, lambdap, epochp, num_f in itertools.product(space["lr"], space["lambdap"], space["epochp"], space["num_f"]):
        rows.append({"lr": lr, "lambdap": lambdap, "epochp": epochp, "num_f": num_f})
    return rows


def sample_random_configs(space: Dict[str, Sequence[float | int]] | None, count: int, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = grid(space)
    rng.shuffle(rows)
    out = []
    for index, config in enumerate(rows[:count]):
        out.append({"proposal_id": f"random_{index:03d}", "hypothesis": "random search proposal", "config": dict(config)})
    return out
