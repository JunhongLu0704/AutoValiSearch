from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.io import write_json, write_jsonl


def save_trace_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    write_jsonl(path, rows)


def save_artifact(path: str | Path, payload: Any) -> None:
    write_json(path, payload)


