from __future__ import annotations

import argparse
import csv
import json
from statistics import pstdev
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from utils.io import write_csv, write_json, write_jsonl
from .protocol_scoring import score_protocol
from .validator_dsl import FINAL_EPOCH_SELECTION_RULE, REQUIRED_SELECTION_RULE, TEST_EPOCH_SELECTION_RULE

VIEW_SCORE_KEYS = {
    "source_val",
    "color_jitter_low",
    "color_jitter_medium",
    "gaussian_blur_low",
    "gaussian_blur_medium",
    "grayscale",
    "noise_low",
    "random_resized_crop_mild",
    "selection_anchor",
}

def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]).upper(), str(row["split"]), str(row["seed"]))].append(dict(row))
    return grouped


def _select_for_group(protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selection_rule = str(protocol.get("selection_rule") or REQUIRED_SELECTION_RULE)
    scored = []
    for row in rows:
        test_acc = _safe_float(row.get("test_acc"))
        if test_acc is None:
            continue
        view_scores = {}
        for key in VIEW_SCORE_KEYS:
            value = _safe_float(row.get(key))
            if value is not None:
                view_scores[key] = value
        protocol_score = score_protocol(protocol, view_scores)
        scored.append({"epoch": int(row["epoch"]), "protocol_score": protocol_score, "test_acc": test_acc, "view_scores": view_scores})
    if not scored:
        raise ValueError("no valid checkpoint rows available for protocol evaluation")
    if selection_rule == FINAL_EPOCH_SELECTION_RULE:
        scored.sort(key=lambda item: item["epoch"])
        selected = scored[-1]
    elif selection_rule == TEST_EPOCH_SELECTION_RULE:
        scored.sort(key=lambda item: (-item["test_acc"], item["epoch"]))
        selected = scored[0]
    else:
        scored.sort(key=lambda item: (-item["protocol_score"], item["epoch"]))
        selected = scored[0]
    oracle = max(scored, key=lambda item: (item["test_acc"], -item["epoch"]))
    top3 = sorted(scored, key=lambda item: (-item["test_acc"], item["epoch"]))[:3]
    top5 = sorted(scored, key=lambda item: (-item["test_acc"], item["epoch"]))[:5]
    return {
        "selected_epoch": int(selected["epoch"]),
        "selected_test_acc": float(selected["test_acc"]),
        "oracle_epoch": int(oracle["epoch"]),
        "oracle_test_acc": float(oracle["test_acc"]),
        "hit_top3": selected["epoch"] in {item["epoch"] for item in top3},
        "hit_top5": selected["epoch"] in {item["epoch"] for item in top5},
        "protocol_scores": [item["protocol_score"] for item in scored],
        "test_scores": [item["test_acc"] for item in scored],
    }


def evaluate_protocols_against_scores(
    protocols: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped = _group_rows(rows)
    result_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    for protocol in protocols:
        group_selected: List[Dict[str, Any]] = []
        group_oracle: List[Dict[str, Any]] = []
        for (group_dataset, split, seed), group_rows in grouped.items():
            if group_dataset != str(dataset).upper():
                continue
            valid_group_rows = [row for row in group_rows if _safe_float(row.get("test_acc")) is not None]
            if not valid_group_rows:
                continue
            selection = _select_for_group(protocol, valid_group_rows)
            group_selected.append(selection)
            group_oracle.append(selection)
            selected_rows.append({"protocol_name": protocol["protocol_name"], "dataset": group_dataset, "split": split, "seed": int(seed), **selection})
        selected_mean = sum(item["selected_test_acc"] for item in group_selected) / len(group_selected) if group_selected else 0.0
        oracle_mean = sum(item["oracle_test_acc"] for item in group_oracle) / len(group_oracle) if group_oracle else 0.0
        result_rows.append(
            {
                "protocol_name": protocol["protocol_name"],
                "selected_checkpoint_test_mean": round(selected_mean, 6),
                "selection_regret": round(oracle_mean - selected_mean, 6),
                "top3_epoch_hit_rate": round(sum(1 for item in group_selected if item["hit_top3"]) / len(group_selected), 6) if group_selected else 0.0,
                "top5_epoch_hit_rate": round(sum(1 for item in group_selected if item["hit_top5"]) / len(group_selected), 6) if group_selected else 0.0,
                "mean_epoch_distance_to_oracle": round(sum(abs(item["selected_epoch"] - item["oracle_epoch"]) for item in group_selected) / len(group_selected), 6) if group_selected else 0.0,
            }
        )

    random_rows = [row for row in result_rows if str(row["protocol_name"]).startswith(f"{str(dataset).lower()}_random_validator")]
    if not random_rows:
        random_summary: dict[str, Any] = {}
    else:
        scores = [float(row.get("selected_checkpoint_test_mean", 0.0)) for row in random_rows]
        scores.sort()
        mid = len(scores) // 2
        random_summary = {
            "random_avg": round(sum(scores) / len(scores), 6),
            "random_median": round(scores[mid] if len(scores) % 2 == 1 else (scores[mid - 1] + scores[mid]) / 2.0, 6),
            "random_best_upper_bound": round(max(scores), 6),
            "random_std": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
        }
    return result_rows, selected_rows, random_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase II-B protocols")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--protocols-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scores_path = Path(args.scores_path)
    protocols_path = Path(args.protocols_path)
    if not protocols_path.exists() and protocols_path.is_dir():
        protocols_path = protocols_path / "phase2b_all_protocols.json"
    rows = _read_rows(scores_path)
    protocols = json.loads(protocols_path.read_text(encoding="utf-8"))
    result_rows, selected_rows, random_summary = evaluate_protocols_against_scores(protocols, rows, dataset=str(args.dataset))
    regret_rows = []
    hit_rows = []
    grouped = _group_rows(rows)
    for protocol in protocols:
        for (dataset, split, seed), group_rows in grouped.items():
            if dataset != str(args.dataset).upper():
                continue
            valid_group_rows = [row for row in group_rows if _safe_float(row.get("test_acc")) is not None]
            if not valid_group_rows:
                continue
            selection = _select_for_group(protocol, valid_group_rows)
            regret_rows.append({"protocol_name": protocol["protocol_name"], "dataset": dataset, "split": split, "seed": int(seed), "selection_regret": round(selection["oracle_test_acc"] - selection["selected_test_acc"], 6)})
            hit_rows.append({"protocol_name": protocol["protocol_name"], "dataset": dataset, "split": split, "seed": int(seed), "top3_epoch_hit": int(bool(selection["hit_top3"])), "top5_epoch_hit": int(bool(selection["hit_top5"]))})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "phase2b_protocol_results.jsonl", result_rows)
    write_csv(output_dir / "phase2b_protocol_results.csv", result_rows, ["protocol_name", "selected_checkpoint_test_mean", "selection_regret", "top3_epoch_hit_rate", "top5_epoch_hit_rate", "mean_epoch_distance_to_oracle"])
    write_csv(output_dir / "phase2b_selected_epoch_table.csv", selected_rows, ["protocol_name", "dataset", "split", "seed", "selected_epoch", "selected_test_acc", "oracle_epoch", "oracle_test_acc", "hit_top3", "hit_top5"])
    write_csv(output_dir / "phase2b_regret_table.csv", regret_rows, ["protocol_name", "dataset", "split", "seed", "selection_regret"])
    write_csv(output_dir / "phase2b_topk_hit_table.csv", hit_rows, ["protocol_name", "dataset", "split", "seed", "top3_epoch_hit", "top5_epoch_hit"])
    summary_payload = {"dataset": args.dataset, "protocol_count": len(protocols), "rows": result_rows}
    if random_summary:
        summary_payload["random_summary"] = random_summary
    write_json(output_dir / "phase2b_summary.json", summary_payload)
    write_csv(output_dir / "phase2b_summary_table.csv", result_rows, ["protocol_name", "selected_checkpoint_test_mean", "selection_regret", "top3_epoch_hit_rate", "top5_epoch_hit_rate", "mean_epoch_distance_to_oracle"])
    if random_summary:
        write_json(output_dir / "phase2b_random_summary.json", random_summary)
    print(json.dumps({"dataset": args.dataset, "protocol_count": len(protocols)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


