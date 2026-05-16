from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentState:
    name: str
    backend: str
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RuntimeState:
    project_name: str
    agents: Dict[str, AgentState]
    artifacts: Dict[str, Any] = field(default_factory=dict)

