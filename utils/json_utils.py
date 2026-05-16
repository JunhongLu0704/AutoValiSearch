from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def redact_secrets(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        redacted = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("api_key", "token", "secret", "password")):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_secrets(value)
        return redacted
    if isinstance(payload, list):
        return [redact_secrets(item) for item in payload]
    return payload

