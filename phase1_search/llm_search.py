from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .search_space import build_phase1_search_space, grid, repair_phase1_config


def _history_keys(history: Sequence[Mapping[str, Any]] | None) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for item in history or []:
        config = item.get("config") or item.get("proposal")
        if isinstance(config, Mapping):
            keys.add((config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f")))
    return keys


def propose_llm(count: int = 8, history: Sequence[Mapping[str, Any]] | None = None) -> List[Dict[str, Any]]:
    rows = grid(build_phase1_search_space())
    rows.sort(
        key=lambda item: (
            abs(float(item["lr"]) - 0.02),
            abs(float(item["lambdap"]) - 4.0),
            abs(int(item["epochp"]) - 4),
            abs(int(item["num_f"]) - 3),
        )
    )
    seen = _history_keys(history)
    proposals: List[Dict[str, Any]] = []
    for index, config in enumerate(rows):
        key = (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))
        if key in seen:
            continue
        proposals.append({"proposal_id": f"llm_{index:03d}", "hypothesis": "LLM-guided proposal", "config": repair_phase1_config(dict(config))})
        if len(proposals) >= count:
            break
    return proposals


def propose_llm_one(history: Sequence[Mapping[str, Any]] | None = None) -> Dict[str, Any]:
    proposals = propose_llm(count=1, history=history)
    if not proposals:
        raise ValueError("No llm proposal available")
    return proposals[0]
