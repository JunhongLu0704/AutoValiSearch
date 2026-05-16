from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class BackendSpec:
    name: str
    base_url: str
    api_key_env: str
    model: str
    timeout_sec: int = 120
    max_retries: int = 2
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "model": self.model,
            "timeout_sec": self.timeout_sec,
            "max_retries": self.max_retries,
            "extra_headers": dict(self.extra_headers),
        }


@dataclass(frozen=True)
class BackendRegistry:
    default_backend: str
    backends: Mapping[str, BackendSpec]

    def resolve(self, name: str | None = None) -> BackendSpec:
        selected = name or self.default_backend
        if selected not in self.backends:
            raise KeyError(f"Unknown backend: {selected}")
        return self.backends[selected]

