from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .search_space import build_phase1_search_space, grid, repair_phase1_config


def _load_optuna():
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError:
        return None
    return optuna


def _history_keys(history: Sequence[Mapping[str, Any]] | None) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for item in history or []:
        config = item.get("config") or item.get("proposal")
        if isinstance(config, Mapping):
            keys.add((config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f")))
    return keys


def _history_score(item: Mapping[str, Any]) -> float | None:
    candidates = [
        item.get("mean_test_acc"),
        item.get("selection_score"),
        item.get("test_score"),
    ]
    result = item.get("result")
    if isinstance(result, Mapping):
        candidates.extend(
            [
                result.get("mean_test_acc"),
                result.get("selection_score"),
                result.get("test_score"),
            ]
        )
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _make_optuna_proposal(
    *,
    history: Sequence[Mapping[str, Any]] | None,
    seed: int,
) -> Dict[str, Any] | None:
    optuna = _load_optuna()
    if optuna is None:
        return None

    space = build_phase1_search_space()
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    distributions = {
        "lr": optuna.distributions.CategoricalDistribution(choices=list(space["lr"])),
        "lambdap": optuna.distributions.CategoricalDistribution(choices=list(space["lambdap"])),
        "epochp": optuna.distributions.CategoricalDistribution(choices=list(space["epochp"])),
        "num_f": optuna.distributions.CategoricalDistribution(choices=list(space["num_f"])),
    }
    seen = _history_keys(history)

    for item in history or []:
        config = item.get("config") or item.get("proposal")
        if not isinstance(config, Mapping):
            continue
        try:
            canonical = repair_phase1_config(dict(config))
        except Exception:
            continue
        score = _history_score(item)
        if score is None:
            continue
        try:
            trial = optuna.trial.create_trial(
                params=dict(canonical),
                distributions=distributions,
                value=float(score),
            )
        except Exception:
            continue
        study.add_trial(trial)

    trial = study.ask(fixed_distributions=distributions)
    config = repair_phase1_config(dict(trial.params))
    key = (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))
    if key in seen:
        return None
    return {
        "proposal_id": "tpe_000",
        "hypothesis": "optuna TPE proposal",
        "config": dict(config),
    }


def propose_tpe(count: int = 8, history: Sequence[Mapping[str, Any]] | None = None) -> List[Dict[str, Any]]:
    seen = _history_keys(history)
    proposals: List[Dict[str, Any]] = []

    optuna = _load_optuna()
    if optuna is None:
        rows = grid(build_phase1_search_space())
        rows.sort(key=lambda item: (abs(float(item["lr"]) - 0.0025), abs(float(item["lambdap"]) - 4.0), int(item["epochp"]), int(item["num_f"])))
        for index, config in enumerate(rows):
            key = (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))
            if key in seen:
                continue
            proposals.append({"proposal_id": f"tpe_{index:03d}", "hypothesis": "heuristic TPE-style proposal", "config": dict(config)})
            if len(proposals) >= count:
                break
        return proposals

    attempts = 0
    seed_base = 0
    while len(proposals) < count and attempts < max(32, count * 8):
        proposal = _make_optuna_proposal(history=history, seed=seed_base + attempts)
        attempts += 1
        if proposal is None:
            continue
        key = (proposal["config"].get("lr"), proposal["config"].get("lambdap"), proposal["config"].get("epochp"), proposal["config"].get("num_f"))
        if key in seen:
            continue
        seen.add(key)
        proposal["proposal_id"] = f"tpe_{len(proposals):03d}"
        proposals.append(proposal)
    if proposals:
        return proposals

    rows = grid(build_phase1_search_space())
    rows.sort(key=lambda item: (abs(float(item["lr"]) - 0.0025), abs(float(item["lambdap"]) - 4.0), int(item["epochp"]), int(item["num_f"])))
    for index, config in enumerate(rows):
        key = (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))
        if key in seen:
            continue
        proposals.append({"proposal_id": f"tpe_{index:03d}", "hypothesis": "heuristic TPE-style proposal", "config": dict(config)})
        if len(proposals) >= count:
            break
    return proposals
