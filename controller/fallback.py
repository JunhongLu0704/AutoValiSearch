from __future__ import annotations

from typing import Any, Callable


def call_with_fallback(primary: Callable[[], Any], fallback: Callable[[], Any]) -> Any:
    try:
        return primary()
    except Exception:
        return fallback()

