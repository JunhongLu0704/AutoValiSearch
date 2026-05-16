from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence

from phase2b_validation.policy_compiler import compile_policy


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset", "")).upper(), str(row.get("split")), str(row.get("seed")))].append(dict(row))
    for values in grouped.values():
        values.sort(key=lambda row: int(row.get("epoch", 0) or 0))
    return grouped


def _baseline_selection(rows: Sequence[Mapping[str, Any]], score_key: str) -> dict[str, Any]:
    scored = [{"epoch": int(row.get("epoch", 0) or 0), "score": float(row.get(score_key, 0.0) or 0.0), "test_acc": float(row.get("test_acc", 0.0) or 0.0)} for row in rows]
    selected = sorted(scored, key=lambda item: (-item["score"], item["epoch"]))[0]
    oracle = max(scored, key=lambda item: (item["test_acc"], -item["epoch"]))
    top3 = sorted(scored, key=lambda item: (-item["test_acc"], item["epoch"]))[:3]
    top5 = sorted(scored, key=lambda item: (-item["test_acc"], item["epoch"]))[:5]
    return {**selected, "oracle_epoch": oracle["epoch"], "oracle_test_acc": oracle["test_acc"], "hit_top3": selected["epoch"] in {item["epoch"] for item in top3}, "hit_top5": selected["epoch"] in {item["epoch"] for item in top5}}


def _normalize(values: list[float], source_values: list[float], normalization: str) -> list[float]:
    if normalization == "raw":
        return values
    if normalization == "rank_within_split_seed":
        order = {index: rank for rank, (index, _) in enumerate(sorted(enumerate(values), key=lambda item: item[1]), start=1)}
        denom = max(len(values) - 1, 1)
        return [(order[index] - 1) / denom for index in range(len(values))]
    if normalization == "zscore_within_split_seed":
        mu = mean(values)
        sd = pstdev(values) or 1.0
        return [(value - mu) / sd for value in values]
    if normalization == "minmax_within_split_seed":
        lo, hi = min(values), max(values)
        span = hi - lo or 1.0
        return [(value - lo) / span for value in values]
    if normalization == "center_by_source_val":
        return [value - source for value, source in zip(values, source_values)]
    if normalization == "softmax_within_split_seed":
        beta = 5.0
        if not values:
            return []
        m = max(values)
        weights = [math.exp(beta * (value - m)) for value in values]
        total = sum(weights) or 1.0
        return [weight / total for weight in weights]
    if normalization == "sigmoid_within_split_seed":
        return [1.0 / (1.0 + math.exp(-value)) for value in values]
    return values


def _aggregate(values: list[float], views: list[str], aggregation: Mapping[str, Any]) -> float:
    agg_type = str(aggregation.get("type") or "mean")
    if not values:
        return 0.0
    if agg_type == "weighted_mean":
        weights = dict(aggregation.get("weights") or {})
        return sum(float(weights.get(view, 0.0)) * value for view, value in zip(views, values))
    if agg_type == "harmonic_mean":
        shifted = [max(value, 1e-6) for value in values]
        return len(shifted) / sum(1.0 / value for value in shifted)
    if agg_type == "geometric_mean":
        shifted = [max(value, 1e-6) for value in values]
        return math.exp(sum(math.log(value) for value in shifted) / len(shifted))
    if agg_type == "mean_minus_std":
        return mean(values) - (pstdev(values) if len(values) > 1 else 0.0)
    if agg_type == "trimmed_mean":
        ordered = sorted(values)
        if len(ordered) > 2:
            ordered = ordered[1:-1]
        return mean(ordered)
    if agg_type == "softmin":
        beta = float(aggregation.get("beta", 5.0) or 5.0)
        m = min(values)
        return -math.log(sum(math.exp(-beta * (value - m)) for value in values)) / beta + m
    if agg_type == "median":
        return float(median(values))
    if agg_type == "max":
        return max(values)
    if agg_type == "logsumexp":
        beta = float(aggregation.get("beta", 1.0) or 1.0)
        m = max(values)
        return math.log(sum(math.exp(beta * (value - m)) for value in values)) / beta + m
    return mean(values)


def _select_epoch(scored: list[dict[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    rule_type = str(rule.get("type") or "argmax_score")
    best_score = max(float(item["policy_score"]) for item in scored)
    epsilon = float(rule.get("epsilon", 0.0) or 0.0)
    candidates = [item for item in scored if float(item["policy_score"]) >= best_score - epsilon]
    if rule_type == "latest_within_epsilon_of_best":
        return max(candidates, key=lambda item: int(item["epoch"]))
    if rule_type == "earliest_within_epsilon_of_best":
        return min(candidates, key=lambda item: int(item["epoch"]))
    if rule_type == "smoothed_score_argmax":
        by_epoch = {int(item["epoch"]): item for item in scored}
        smoothed = []
        for item in scored:
            epoch = int(item["epoch"])
            neighbors = [float(by_epoch[e]["policy_score"]) for e in [epoch - 1, epoch, epoch + 1] if e in by_epoch]
            smoothed.append({**item, "policy_score": mean(neighbors)})
        return max(smoothed, key=lambda item: (float(item["policy_score"]), int(item["epoch"])))
    tie_break = str(rule.get("tie_break") or "later_epoch")
    return max(candidates, key=lambda item: (float(item["policy_score"]), int(item["epoch"]) if tie_break == "later_epoch" else -int(item["epoch"])))


def _view_lookup(view_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, int, str], float]:
    lookup: dict[tuple[str, str, str, int, str], float] = {}
    for row in view_rows:
        lookup[(str(row.get("dataset", "")).upper(), str(row.get("split")), str(row.get("seed")), int(row.get("epoch", 0) or 0), str(row.get("view_name")))] = float(row.get("val_score", 0.0) or 0.0)
    return lookup


def evaluate_policy(
    policy: Mapping[str, Any],
    score_rows: Sequence[Mapping[str, Any]],
    view_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    dataset: str,
    run_idx: int | None = None,
    round_idx: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    compiled = compile_policy(policy)
    views = list(compiled["policy"]["views"])
    grouped = _group(score_rows)
    lookup = _view_lookup(view_rows or [])
    selections: list[dict[str, Any]] = []
    vanilla: list[dict[str, Any]] = []
    upper: list[dict[str, Any]] = []
    for (group_dataset, split, seed), rows in grouped.items():
        if group_dataset != dataset.upper():
            continue
        source_values = [float(row.get("source_val", 0.0) or 0.0) for row in rows]
        per_view_values: dict[str, list[float]] = {}
        for view in views:
            if view == "source_val":
                per_view_values[view] = source_values
            else:
                per_view_values[view] = [
                    lookup.get((group_dataset, split, seed, int(row.get("epoch", 0) or 0), view), float(row.get("source_val", 0.0) or 0.0))
                    for row in rows
                ]
        normed = {view: _normalize(values, source_values, compiled["policy"]["normalization"]) for view, values in per_view_values.items()}
        scored = []
        for index, row in enumerate(rows):
            values = [normed[view][index] for view in views]
            scored.append({"epoch": int(row.get("epoch", 0) or 0), "policy_score": _aggregate(values, views, compiled["policy"]["aggregation"]), "test_acc": float(row.get("test_acc", 0.0) or 0.0), "checkpoint_id": row.get("checkpoint_id")})
        selected = _select_epoch(scored, compiled["policy"]["epoch_selection"])
        oracle = max(scored, key=lambda item: (item["test_acc"], -item["epoch"]))
        top3 = sorted(scored, key=lambda item: (-item["test_acc"], item["epoch"]))[:3]
        top5 = sorted(scored, key=lambda item: (-item["test_acc"], item["epoch"]))[:5]
        selections.append({"dataset": group_dataset, "split": split, "seed": int(seed), "selected_epoch": int(selected["epoch"]), "selected_test_acc": float(selected["test_acc"]), "oracle_epoch": int(oracle["epoch"]), "oracle_test_acc": float(oracle["test_acc"]), "hit_top3": selected["epoch"] in {item["epoch"] for item in top3}, "hit_top5": selected["epoch"] in {item["epoch"] for item in top5}})
        vanilla.append(_baseline_selection(rows, "source_val"))
        upper.append(_baseline_selection(rows, "test_acc"))
    selected_mean = mean([item["selected_test_acc"] for item in selections]) if selections else 0.0
    vanilla_mean = mean([item["test_acc"] for item in vanilla]) if vanilla else 0.0
    upper_mean = mean([item["test_acc"] for item in upper]) if upper else 0.0
    raw_selected_mean = selected_mean
    used_fallback = False
    safety = compiled["policy"]["safety_rule"]
    if safety.get("compare_with_vanilla") and safety.get("fallback_to_vanilla_if_underperform") and selected_mean < vanilla_mean:
        selected_mean = vanilla_mean
        used_fallback = True
        selections = [
            {
                **selection,
                "selected_epoch": int(vanilla_item["epoch"]),
                "selected_test_acc": float(vanilla_item["test_acc"]),
                "hit_top3": bool(vanilla_item["hit_top3"]),
                "hit_top5": bool(vanilla_item["hit_top5"]),
                "used_safety_fallback": True,
            }
            for selection, vanilla_item in zip(selections, vanilla)
        ]
    oracle_mean = mean([item["oracle_test_acc"] for item in selections]) if selections else 0.0
    metrics = {
        "policy_name": compiled["policy_name"],
        "protocol_name": compiled["policy_name"],
        "run_idx": run_idx,
        "round_idx": round_idx,
        "selected_checkpoint_test_mean": round(raw_selected_mean, 6),
        "deployable_selected_test_mean": round(selected_mean, 6),
        "vanilla_best_val": round(vanilla_mean, 6),
        "improvement_over_vanilla": round(raw_selected_mean - vanilla_mean, 6),
        "deployable_improvement_over_vanilla": round(selected_mean - vanilla_mean, 6),
        "selection_regret": round(max(upper_mean, oracle_mean) - selected_mean, 6),
        "top3_epoch_hit_rate": round(sum(1 for item in selections if item["hit_top3"]) / len(selections), 6) if selections else 0.0,
        "top5_epoch_hit_rate": round(sum(1 for item in selections if item["hit_top5"]) / len(selections), 6) if selections else 0.0,
        "gap_to_best_test_upper_bound": round(upper_mean - selected_mean, 6),
        "selected_epoch_mean": round(mean([item["selected_epoch"] for item in selections]), 6) if selections else 0.0,
        "selected_epoch_std": round(pstdev([item["selected_epoch"] for item in selections]), 6) if len(selections) > 1 else 0.0,
        "used_safety_fallback": used_fallback,
    }
    return metrics, selections
