from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from llm.json_response import loads_json
from llm.openai_compatible_client import OpenAICompatibleClient
from llm.prompts import validation_designer_messages
from llm.schemas import BackendSpec
from llm.token_budget import resolve_max_tokens
from phase2b_validation.policy_compiler import compile_policy
from phase2b_validation.run_val_designer import (
    _fallback_policy,
    _extract_policy_payload,
    build_policy_initialization_prompt,
    design_protocol_round,
    design_protocol_rounds,
)
from phase2b_validation.val_design_memory import summarize_val_design_memory
from phase2b_validation.validator_dsl import compile_protocol
from utils.io import append_jsonl, write_json, write_jsonl


def _history_protocol_names(history: list[Mapping[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for item in history or []:
        name = str(item.get("protocol_name") or "").strip()
        if name:
            names.add(name)
    return names


@dataclass
class ValDesignerAgent:
    backend: BackendSpec | None = None
    trace_dir: Path | None = None
    max_tokens: int | None = None

    def propose_policy(
        self,
        brief: Mapping[str, Any],
        *,
        run_index: int,
        round_index: int,
        memory_summary: Mapping[str, Any] | None = None,
        previous_feedback: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset = str(brief.get("dataset", "PACS")).upper()
        fallback = compile_policy(_fallback_policy(dataset, run_index, round_index))
        if self.backend is None:
            return fallback, {"fallback_used": True, "json_valid": True, "repair_attempted": False}
        client = OpenAICompatibleClient(self.backend, trace_dir=(self.trace_dir / "llm_trace") if self.trace_dir is not None else None)
        messages = build_policy_initialization_prompt(brief, dict(memory_summary or {}), previous_feedback)
        max_tokens = int(self.max_tokens or resolve_max_tokens(specific_env="VAL_DESIGNER_MAX_TOKENS", default=8192))
        repair_max_tokens = int(resolve_max_tokens(specific_env="VAL_DESIGNER_REPAIR_MAX_TOKENS", default=max_tokens))
        try:
            if self.trace_dir is not None:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                append_jsonl(
                    self.trace_dir / "llm_trace" / "llm_raw_prompts.jsonl",
                    {"run_index": run_index, "round_index": round_index, "messages": [message.__dict__ for message in messages], "max_tokens": max_tokens},
                )
            response = client.chat(messages, temperature=0.2, max_tokens=max_tokens)
            finish_reason = response.get("choices", [{}])[0].get("finish_reason") if isinstance(response.get("choices"), list) and response.get("choices") else None
            response_truncated = finish_reason == "length"
            if self.trace_dir is not None:
                append_jsonl(
                    self.trace_dir / "llm_trace" / "llm_raw_responses.jsonl",
                    {
                        "run_index": run_index,
                        "round_index": round_index,
                        "response": response,
                        "finish_reason": finish_reason,
                        "llm_response_truncated": response_truncated,
                    },
                )
            parsed = loads_json(response["choices"][0]["message"]["content"])
            return compile_policy(_extract_policy_payload(parsed)), {
                "fallback_used": False,
                "json_valid": True,
                "repair_attempted": False,
                "max_tokens": max_tokens,
                "finish_reason": finish_reason,
                "llm_response_truncated": response_truncated,
            }
        except Exception as exc:
            repair_messages = messages + [
                type(messages[0])(
                    role="user",
                    content=json.dumps(
                        {
                            "task": "repair the policy JSON",
                            "validation_error": str(exc),
                            "previous_feedback": dict(previous_feedback or {}),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            ]
            try:
                if self.trace_dir is not None:
                    append_jsonl(
                        self.trace_dir / "llm_trace" / "llm_validation_errors.jsonl",
                        {
                            "run_index": run_index,
                            "round_index": round_index,
                            "validation_error": str(exc),
                            "max_tokens": max_tokens,
                        },
                    )
                    append_jsonl(
                        self.trace_dir / "llm_trace" / "llm_repair_prompts.jsonl",
                        {"run_index": run_index, "round_index": round_index, "messages": [message.__dict__ for message in repair_messages], "repair_max_tokens": repair_max_tokens},
                    )
                response = client.chat(repair_messages, temperature=0.1, max_tokens=repair_max_tokens)
                repair_finish_reason = response.get("choices", [{}])[0].get("finish_reason") if isinstance(response.get("choices"), list) and response.get("choices") else None
                repair_response_truncated = repair_finish_reason == "length"
                if self.trace_dir is not None:
                    append_jsonl(
                        self.trace_dir / "llm_trace" / "llm_repair_responses.jsonl",
                        {
                            "run_index": run_index,
                            "round_index": round_index,
                            "response": response,
                            "finish_reason": repair_finish_reason,
                            "llm_response_truncated": repair_response_truncated,
                        },
                    )
                parsed = loads_json(response["choices"][0]["message"]["content"])
                return compile_policy(_extract_policy_payload(parsed)), {
                    "fallback_used": False,
                    "json_valid": True,
                    "repair_attempted": True,
                    "max_tokens": max_tokens,
                    "repair_max_tokens": repair_max_tokens,
                    "finish_reason": finish_reason,
                    "repair_finish_reason": repair_finish_reason,
                    "llm_response_truncated": response_truncated,
                    "repair_llm_response_truncated": repair_response_truncated,
                }
            except Exception as repair_exc:
                if self.trace_dir is not None:
                    self.trace_dir.mkdir(parents=True, exist_ok=True)
                    append_jsonl(
                        self.trace_dir / "llm_trace" / "llm_repair_errors.jsonl",
                        {
                            "run_index": run_index,
                            "round_index": round_index,
                            "validation_error": str(exc),
                            "repair_error": str(repair_exc),
                            "max_tokens": max_tokens,
                            "repair_max_tokens": repair_max_tokens,
                        },
                    )
                    write_json(self.trace_dir / "policy_fallback_usage.json", {"fallback_used": True, "error": str(repair_exc)})
                return fallback, {
                    "fallback_used": True,
                    "json_valid": False,
                    "repair_attempted": True,
                    "error": str(repair_exc),
                    "max_tokens": max_tokens,
                    "repair_max_tokens": repair_max_tokens,
                }

    def design(
        self,
        brief: Mapping[str, Any],
        *,
        num_runs: int = 3,
        rounds: int = 12,
        history: list[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        protocols: list[dict[str, Any]] = []
        external_history = list(history or [])
        max_tokens = int(self.max_tokens or resolve_max_tokens(specific_env="VAL_DESIGNER_MAX_TOKENS", default=8192))
        repair_max_tokens = int(resolve_max_tokens(specific_env="VAL_DESIGNER_REPAIR_MAX_TOKENS", default=max_tokens))
        if self.backend is not None:
            client = OpenAICompatibleClient(self.backend, trace_dir=(self.trace_dir / "llm_trace") if self.trace_dir is not None else None)
            for run_index in range(int(num_runs)):
                run_history: list[Mapping[str, Any]] = []
                for round_index in range(int(rounds)):
                    seen_names = _history_protocol_names(external_history + run_history)
                    messages = validation_designer_messages(
                        brief,
                        num_runs,
                        rounds,
                        history=external_history + run_history,
                        run_index=run_index,
                        round_index=round_index,
                        memory_summary=summarize_val_design_memory([{"protocols": external_history + run_history}], []),
                    )
                    if self.trace_dir is not None:
                        self.trace_dir.mkdir(parents=True, exist_ok=True)
                        append_jsonl(
                            self.trace_dir / "llm_trace" / "llm_raw_prompts.jsonl",
                            {"run_index": run_index, "round_index": round_index, "messages": [message.__dict__ for message in messages], "max_tokens": max_tokens},
                        )
                    try:
                        response = client.chat(messages, temperature=0.2, max_tokens=max_tokens)
                        content = response["choices"][0]["message"]["content"]
                        if self.trace_dir is not None:
                            finish_reason = response.get("choices", [{}])[0].get("finish_reason") if isinstance(response.get("choices"), list) and response.get("choices") else None
                            append_jsonl(
                                self.trace_dir / "llm_trace" / "llm_raw_responses.jsonl",
                                {
                                    "run_index": run_index,
                                    "round_index": round_index,
                                    "response": response,
                                    "finish_reason": finish_reason,
                                    "llm_response_truncated": finish_reason == "length",
                                },
                            )
                        parsed = loads_json(content)
                        raw_protocol = parsed.get("protocol")
                        if raw_protocol is None and isinstance(parsed.get("protocols"), list) and parsed["protocols"]:
                            raw_protocol = parsed["protocols"][0]
                        if isinstance(raw_protocol, dict):
                            protocol_name = str(raw_protocol.get("protocol_name") or "").strip()
                            if protocol_name and protocol_name in seen_names:
                                raise ValueError("duplicate protocol proposal")
                            compiled = compile_protocol(raw_protocol)
                            protocols.append(compiled)
                            run_history.append(compiled)
                            continue
                        if self.trace_dir is not None:
                            append_jsonl(
                                self.trace_dir / "llm_trace" / "llm_validation_errors.jsonl",
                                {"error": "llm response missing valid protocol", "run_index": run_index, "round_index": round_index},
                            )
                    except Exception:
                        if self.trace_dir is not None:
                            append_jsonl(
                                self.trace_dir / "llm_trace" / "llm_validation_errors.jsonl",
                                {"error": "llm parse or validation failed", "run_index": run_index, "round_index": round_index},
                            )
                        pass
                    fallback = design_protocol_round(brief, run_index=run_index, round_index=round_index, history=external_history + run_history)
                    if self.trace_dir is not None:
                        append_jsonl(
                            self.trace_dir / "llm_trace" / "llm_fallback_usage.jsonl",
                            {"fallback_used": True, "run_index": run_index, "round_index": round_index},
                        )
                    protocols.append(fallback)
                    run_history.append(fallback)
            return protocols
        return design_protocol_rounds(dict(brief), num_runs=num_runs, rounds=rounds, history=external_history)

    def save_design(self, path: str | Path, brief: Mapping[str, Any], *, num_runs: int = 3, rounds: int = 12) -> None:
        protocols = self.design(brief, num_runs=num_runs, rounds=rounds)
        write_json(path, {"agent": "Val Designer Agent", "protocol_count": len(protocols), "backend": self.backend.public_dict() if self.backend else None})
        write_jsonl(Path(path).with_suffix(".jsonl"), protocols)

