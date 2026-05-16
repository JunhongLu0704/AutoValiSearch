from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from reporting.report_agent import generate_report_bundle


@dataclass
class ReportAgent:
    backend_name: str = "offline"

    def write(self, evidence_pack: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
        return generate_report_bundle(evidence_pack, output_dir=Path(output_dir))


