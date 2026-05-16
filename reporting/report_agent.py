from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from llm.backend_config import load_backend_choice
from llm.schemas import BackendSpec
from llm.token_budget import resolve_max_tokens
from utils.io import read_json, write_json, write_jsonl, write_text

from .claim_audit import audit_claims
from .prompts import build_report_prompt_pack
from .render_markdown import render_demo_report, render_limitations, render_ppt_outline, render_readme_summary, render_resume_snippet


def _fallback_outputs(evidence_pack: Mapping[str, Any]) -> dict[str, str]:
    return {
        "demo_report": render_demo_report(evidence_pack),
        "readme_summary": render_readme_summary(evidence_pack),
        "resume_snippet": render_resume_snippet(evidence_pack),
        "ppt_outline": render_ppt_outline(evidence_pack),
        "limitations": render_limitations(evidence_pack),
    }


def generate_report_bundle(evidence_pack: Mapping[str, Any], output_dir: Path, backend: BackendSpec | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "demo_report": output_dir / "demo_report.md",
        "readme_summary": output_dir / "readme_summary.md",
        "resume_snippet": output_dir / "resume_snippet.md",
        "ppt_outline": output_dir / "ppt_outline.md",
        "limitations": output_dir / "limitations.md",
        "claim_audit": output_dir / "claim_audit.json",
        "report_agent_trace": output_dir / "report_agent_trace.jsonl",
        "report_agent_prompt_pack": output_dir / "report_agent_prompt_pack.json",
    }
    prompt_pack = build_report_prompt_pack(evidence_pack)
    write_json(outputs["report_agent_prompt_pack"], prompt_pack)

    max_tokens = resolve_max_tokens(specific_env="REPORT_AGENT_MAX_TOKENS", default=8192)
    rendered = _fallback_outputs(evidence_pack)
    trace_rows: list[dict[str, Any]] = [
        {"type": "system_prompt", "content": "deterministic_evidence_reporter"},
        {
            "type": "status",
            "content": "deterministic_template_generation",
            "backend_name": backend.name if backend else "deterministic_evidence_reporter",
            "model": backend.model if backend else "offline-template",
            "base_url_redacted": backend.base_url if backend else None,
            "latency_sec": 0.0,
            "max_tokens": max_tokens,
            "finish_reason": "stop",
            "json_valid": True,
            "fallback_used": False,
            "llm_generation_used": False,
        },
    ]
    write_jsonl(output_dir / "llm_trace" / "llm_trace.jsonl", trace_rows)

    write_text(outputs["demo_report"], rendered["demo_report"])
    write_text(outputs["readme_summary"], rendered["readme_summary"])
    write_text(outputs["resume_snippet"], rendered["resume_snippet"])
    write_text(outputs["ppt_outline"], rendered["ppt_outline"])
    write_text(outputs["limitations"], rendered["limitations"])

    claim_audit = audit_claims(evidence_pack, report_texts=rendered)
    write_json(outputs["claim_audit"], claim_audit)
    trace_rows.append({"type": "claim_audit", "payload": claim_audit})
    write_jsonl(outputs["report_agent_trace"], trace_rows)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report outputs from an evidence pack")
    parser.add_argument("--evidence-pack", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend-config")
    parser.add_argument("--backend-name")
    parser.add_argument("--backend", choices=["cloud", "local_openai_compatible"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pack = read_json(args.evidence_pack)
    backend_name = args.backend_name or args.backend
    backend = load_backend_choice(args.backend_config, backend_name) if args.backend_config else None
    generate_report_bundle(pack, Path(args.output_dir), backend=backend)


if __name__ == "__main__":
    main()
