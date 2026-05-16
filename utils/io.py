from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_jsonl(path: str | Path) -> List[Any]:
    rows: List[Any] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], headers: Iterable[str]) -> None:
    path = Path(path)
    ensure_parent(path)
    headers = list(headers)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})
