from __future__ import annotations

import json
from typing import Any, Mapping

from llm.schemas import LLMMessage

REPORT_AGENT_SYSTEM_PROMPT = "You are Report Agent for AutoValiSearch. Use only the evidence pack."
REPORT_AGENT_USER_PROMPT = "Summarize the evidence pack for README, resume, PPT, limitations, and claim audit. Do not invent results."


def report_agent_messages(evidence_pack: Mapping[str, Any]) -> list[LLMMessage]:
    payload = {
        "task": "generate evidence-based research report outputs",
        "evidence_pack": dict(evidence_pack),
        "output_schema": {
            "demo_report": "markdown string",
            "readme_summary": "markdown string",
            "resume_snippet": "markdown string",
            "ppt_outline": "markdown string",
            "limitations": "markdown string",
        },
        "constraints": [
            "Return JSON only.",
            "Do not invent numbers.",
            "Use only the evidence pack.",
            "Do not claim random best-of-k is a deployable baseline.",
            "Keep the report concise and presentation-ready.",
        ],
    }
    return [
        LLMMessage(role="system", content=REPORT_AGENT_SYSTEM_PROMPT),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def build_report_prompt_pack(evidence_pack: Mapping[str, Any]) -> dict[str, Any]:
    messages = report_agent_messages(evidence_pack)
    return {
        "system_prompt": messages[0].content,
        "user_prompt": messages[1].content,
        "message_count": len(messages),
        "output_schema": {
            "demo_report": "markdown string",
            "readme_summary": "markdown string",
            "resume_snippet": "markdown string",
            "ppt_outline": "markdown string",
            "limitations": "markdown string",
        },
        "claims": {
            "allowed_claims": list(evidence_pack.get("allowed_claims", [])),
            "not_allowed_claims": list(evidence_pack.get("not_allowed_claims", [])),
        },
    }
