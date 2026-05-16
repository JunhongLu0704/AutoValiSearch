from __future__ import annotations

import json
from typing import Any


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def extract_json_text(text: str) -> str:
    text = _strip_code_fences(text)
    if not text:
        raise ValueError("Empty response")
    first_obj = text.find("{")
    first_arr = text.find("[")
    if first_obj == -1 and first_arr == -1:
        raise ValueError("No JSON object found")
    if first_arr != -1 and (first_obj == -1 or first_arr < first_obj):
        start = first_arr
        end = text.rfind("]")
    else:
        start = first_obj
        end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Malformed JSON fragment")
    return text[start : end + 1]


def loads_json(text: str) -> Any:
    return json.loads(extract_json_text(text))
