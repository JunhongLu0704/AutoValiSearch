from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .search_space import build_phase1_search_space, sample_random_configs


def _history_keys(history: Sequence[Mapping[str, Any]] | None) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for item in history or []:
        config = item.get("config") or item.get("proposal")
        if isinstance(config, Mapping):
            keys.add((config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f")))
    return keys


def propose_random(count: int = 8, seed: int = 0, history: Sequence[Mapping[str, Any]] | None = None) -> List[Dict[str, Any]]:
    proposals = sample_random_configs(build_phase1_search_space(), count=count + len(history or []), seed=seed)
    seen = _history_keys(history)
    out: List[Dict[str, Any]] = []
    for item in proposals:
        config = item["config"]
        key = (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))
        if key in seen:
            continue
        out.append(item)
        if len(out) >= count:
            break
    return out
