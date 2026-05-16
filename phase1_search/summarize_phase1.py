from __future__ import annotations

import argparse
from pathlib import Path

from utils.io import read_json, write_text
from utils.tables import markdown_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Phase I results")
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = read_json(args.summary_path)
    rows = payload.get("methods", [])
    text = "# Phase I Summary\n\n" + markdown_table(["method", "best_mean_test_acc", "Best@24"], rows)
    write_text(args.output_path, text)


if __name__ == "__main__":
    main()


