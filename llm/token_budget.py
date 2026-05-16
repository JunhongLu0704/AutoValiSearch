from __future__ import annotations

import os


def resolve_max_tokens(*, specific_env: str, default: int) -> int:
    for env_name in (specific_env, "AGENT_MAX_TOKENS"):
        raw = os.environ.get(env_name)
        if raw is None or not str(raw).strip():
            continue
        try:
            value = int(str(raw))
        except ValueError as exc:
            raise ValueError(f"Invalid token budget in {env_name}: {raw}") from exc
        if value <= 0:
            raise ValueError(f"Token budget must be positive in {env_name}: {raw}")
        return value
    return int(default)
