from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import BackendRegistry, BackendSpec


def _parse_config_text(text: str) -> Mapping[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - fallback path
            raise RuntimeError("YAML parsing requires PyYAML or JSON-compatible config text") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, Mapping):
            raise ValueError("Backend config must be a mapping")
        return loaded


def load_backend_registry(path: str | Path) -> BackendRegistry:
    raw = _parse_config_text(Path(path).read_text(encoding="utf-8"))
    default_backend = str(raw.get("default_backend", "")).strip()
    backends_raw = raw.get("backends") or {}
    if not default_backend:
        raise ValueError("default_backend is required")
    if not isinstance(backends_raw, Mapping):
        raise ValueError("backends must be a mapping")
    backends = {}
    for name, item in backends_raw.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"backend {name} must be a mapping")
        backends[str(name)] = BackendSpec(
            name=str(name),
            base_url=str(item["base_url"]),
            api_key_env=str(item["api_key_env"]),
            model=str(item["model"]),
            timeout_sec=int(item.get("timeout_sec", 120)),
            max_retries=int(item.get("max_retries", 2)),
            extra_headers=dict(item.get("extra_headers") or {}),
        )
    return BackendRegistry(default_backend=default_backend, backends=backends)


def load_backend_choice(path: str | Path, choice: str | None = None) -> BackendSpec:
    registry = load_backend_registry(path)
    return registry.resolve(choice)

