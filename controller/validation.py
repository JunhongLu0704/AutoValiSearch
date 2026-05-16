from __future__ import annotations

from typing import Any, Mapping, Sequence


def require_keys(payload: Mapping[str, Any], keys: Sequence[str], *, context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")


def ensure_no_private_paths(text: str) -> None:
    lowered = text.lower()
    banned = ["c:/users/", "c:\\users\\", "/home/shared/work/", "api_key", "token", "password"]
    for token in banned:
        if token in lowered:
            raise ValueError(f"Private path or secret-like token detected: {token}")

