from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

REQUIRED_TOP_LEVEL_KEYS = {"project", "architecture", "phase1", "phase2b", "allowed_claims", "not_allowed_claims"}


@dataclass(frozen=True)
class ReportEvidencePack:
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        validate_evidence_pack(self.payload)
        return dict(self.payload)


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def validate_evidence_pack(pack: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - set(pack.keys())
    if missing:
        raise ValueError(f"Evidence pack missing keys: {sorted(missing)}")
    if not isinstance(pack["allowed_claims"], list) or not isinstance(pack["not_allowed_claims"], list):
        raise ValueError("allowed_claims and not_allowed_claims must be lists")

    project = _require_mapping("project", pack["project"])
    for key in ["name", "version", "description"]:
        if not project.get(key):
            raise ValueError(f"project.{key} is required")

    architecture = _require_mapping("architecture", pack["architecture"])
    agents = _require_list("architecture.agents", architecture.get("agents"))
    if not agents:
        raise ValueError("architecture.agents must not be empty")
    for agent in agents:
        if isinstance(agent, Mapping):
            if not agent.get("name"):
                raise ValueError("architecture.agents entries must include name when using object form")
        elif not isinstance(agent, str):
            raise ValueError("architecture.agents entries must be strings or mappings")
    if "controller_features" not in architecture or not isinstance(architecture["controller_features"], list):
        raise ValueError("architecture.controller_features must be a list")
    backend_support = architecture.get("backend_support")
    if backend_support is not None and not isinstance(backend_support, list):
        raise ValueError("architecture.backend_support must be a list when provided")

    for phase_name in ["phase1", "phase2b"]:
        phase = _require_mapping(phase_name, pack[phase_name])
        if "metrics" not in phase or not isinstance(phase["metrics"], Mapping):
            raise ValueError(f"{phase_name}.metrics must be a mapping")

    if "backend_comparison" in pack and pack["backend_comparison"] is not None:
        _require_mapping("backend_comparison", pack["backend_comparison"])
    if "agent_traces" in pack and pack["agent_traces"] is not None:
        _require_mapping("agent_traces", pack["agent_traces"])
