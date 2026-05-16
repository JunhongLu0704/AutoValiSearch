from __future__ import annotations

import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm.json_response import loads_json
from llm.openai_compatible_client import OpenAICompatibleClient
from llm.prompts import search_agent_messages
from llm.schemas import BackendSpec
from llm.token_budget import resolve_max_tokens
from phase1_search.search_space import build_phase1_search_space, repair_phase1_config, validate_phase1_config
from phase1_search.llm_search import propose_llm
from phase1_search.random_search import propose_random
from phase1_search.tpe_search import propose_tpe
from phase1_search.search_memory import summarize_search_memory
from utils.io import append_jsonl, write_json


def _coerce_config_value(space_value: list[Any], value: Any) -> Any:
    template = space_value[0] if space_value else value
    if isinstance(template, int) and not isinstance(template, bool):
        return int(float(value))
    if isinstance(template, float):
        return float(value)
    return value


def _history_config_keys(history: Sequence[Mapping[str, Any]] | None) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for item in history or []:
        config = item.get("config") or item.get("proposal")
        if isinstance(config, Mapping):
            keys.add((config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f")))
    return keys


def _config_key(config: Mapping[str, Any]) -> tuple[Any, ...]:
    return (config.get("lr"), config.get("lambdap"), config.get("epochp"), config.get("num_f"))


def _trace_dir(root: Path | None) -> Path | None:
    return root if root is not None else None


def _append_trace(root: Path | None, name: str, row: Mapping[str, Any]) -> None:
    if root is not None:
        append_jsonl(root / name, row)


def _candidate_payloads(parsed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = parsed.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, Mapping)]
    proposals = parsed.get("proposals")
    if isinstance(proposals, list):
        return [item for item in proposals if isinstance(item, Mapping)]
    proposal = parsed.get("proposal")
    if isinstance(proposal, Mapping):
        return [{"proposal": proposal}]
    return []


def _normalize_candidate(item: Mapping[str, Any], *, index: int, seen: set[tuple[Any, ...]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_config = item.get("proposal") or item.get("config") or item
    if not isinstance(raw_config, Mapping):
        return None, {"error_type": "schema_error", "message": "candidate missing proposal/config object", "candidate_index": index}
    try:
        config = validate_phase1_config(raw_config)
    except Exception as exc:
        return None, {"error_type": "grid_error", "message": str(exc), "candidate_index": index, "candidate": dict(raw_config)}
    if _config_key(config) in seen:
        return None, {"error_type": "duplicate_config", "duplicate_config": config, "candidate_index": index}
    return {
        "proposal_id": str(item.get("proposal_id") or f"llm_{index:03d}"),
        "hypothesis": str(item.get("hypothesis") or item.get("design_hypothesis") or "LLM-guided proposal"),
        "design_hypothesis": str(item.get("design_hypothesis") or item.get("hypothesis") or "LLM-guided proposal"),
        "relation_to_memory": str(item.get("relation_to_memory") or "iterative refinement from memory summary"),
        "risk_note": str(item.get("risk_note") or "validated against search space and history"),
        "config": config,
    }, None


@dataclass
class SearchAgent:
    backend: BackendSpec | None = None
    strategy: str = "random"
    seed: int = 0
    trace_dir: Path | None = None
    max_tokens: int | None = None
    method: str | None = None
    dataset: str | None = None
    trial_index: int | None = None

    def propose_one(self, *, history: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        proposals = self.propose(count=1, history=history)
        if not proposals:
            raise ValueError("SearchAgent failed to produce a proposal")
        return proposals[0]

    def propose(self, *, count: int = 24, history: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
        space = build_phase1_search_space()
        seen = _history_config_keys(history)
        max_tokens = int(self.max_tokens or resolve_max_tokens(specific_env="SEARCH_AGENT_MAX_TOKENS", default=8192))
        if self.backend is not None and self.strategy == "llm":
            return self._propose_llm_first(count=count, history=history, seen=seen, max_tokens=max_tokens)
        if self.strategy == "random":
            return propose_random(count=count, seed=self.seed + len(history or []), history=history)
        if self.strategy == "llm":
            return propose_llm(count=count, history=history)
        rng = random.Random(self.seed)
        proposals = propose_tpe(count=count, history=history)
        proposals.sort(key=lambda item: (item["config"]["lr"], item["config"]["lambdap"], item["config"]["epochp"], item["config"]["num_f"]))
        rng.shuffle(proposals)
        return proposals[:count]

    def _propose_llm_first(
        self,
        *,
        count: int,
        history: Sequence[Mapping[str, Any]] | None,
        seen: set[tuple[Any, ...]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        trace_root = _trace_dir(self.trace_dir)
        if trace_root is not None:
            trace_root.mkdir(parents=True, exist_ok=True)
        client = OpenAICompatibleClient(self.backend, trace_dir=None)
        candidates_per_call = int(os.environ.get("SEARCH_AGENT_CANDIDATES_PER_CALL", "3") or 3)
        max_repair_retries = int(os.environ.get("SEARCH_AGENT_MAX_REPAIR_RETRIES", "3") or 3)
        repair_max_tokens = int(os.environ.get("SEARCH_AGENT_REPAIR_MAX_TOKENS", str(max_tokens)) or max_tokens)
        memory = summarize_search_memory(history or [], budget=max(count, len(history or []) + count))
        repair_context: dict[str, Any] | None = None
        initial_json_valid = False
        validation_errors: list[dict[str, Any]] = []
        llm_called = False
        llm_request_ok = False
        finish_reason = None
        response_truncated = False
        candidate_count = 0
        valid_candidate_count = 0
        accepted_candidate_index: int | None = None
        proposal_source = "llm"

        for attempt in range(max_repair_retries + 1):
            messages = search_agent_messages(
                build_phase1_search_space(),
                count,
                history,
                memory_summary=memory,
                candidates_per_call=candidates_per_call,
                repair_context=repair_context,
            )
            _append_trace(trace_root, "llm_raw_prompts.jsonl", {"attempt": attempt, "messages": [message.__dict__ for message in messages]})
            try:
                llm_called = True
                response = client.chat(messages, temperature=0.2 if attempt == 0 else 0.1, max_tokens=max_tokens if attempt == 0 else repair_max_tokens)
                llm_request_ok = True
                finish_reason = response.get("choices", [{}])[0].get("finish_reason") if isinstance(response.get("choices"), list) and response.get("choices") else None
                response_truncated = finish_reason == "length"
                content = response["choices"][0]["message"]["content"]
                _append_trace(trace_root, "llm_raw_responses.jsonl", {"attempt": attempt, "response": response, "finish_reason": finish_reason, "llm_response_truncated": response_truncated})
                parsed = loads_json(content)
                if attempt == 0:
                    initial_json_valid = True
                candidates = _candidate_payloads(parsed)
                candidate_count = len(candidates)
            except Exception as exc:
                error = {"error_type": "json_parse_error", "message": str(exc), "attempt": attempt, "required_schema": "JSON with candidates[].proposal"}
                validation_errors.append(error)
                _append_trace(trace_root, "llm_validation_errors.jsonl", error)
                repair_context = error
                continue

            clean: list[dict[str, Any]] = []
            for index, item in enumerate(candidates):
                proposal, error = _normalize_candidate(item, index=index, seen=seen)
                if error is not None:
                    validation_errors.append({**error, "attempt": attempt})
                    _append_trace(trace_root, "llm_validation_errors.jsonl", {**error, "attempt": attempt})
                    continue
                assert proposal is not None
                clean.append(proposal)
                if accepted_candidate_index is None:
                    accepted_candidate_index = index
                if len(clean) >= count:
                    break
            valid_candidate_count = len(clean)
            if clean:
                if attempt > 0:
                    proposal_source = "repair"
                    _append_trace(trace_root, "llm_repair_attempts.jsonl", {"attempt": attempt, "accepted_candidate_index": accepted_candidate_index, "valid_candidate_count": valid_candidate_count})
                out = clean[:count]
                for proposal in out:
                    proposal["proposal_source"] = proposal_source
                _append_trace(
                    trace_root,
                    "llm_proposal_trace.jsonl",
                    {
                        "proposal_source": proposal_source,
                        "trial_index": self.trial_index,
                        "method": self.method,
                        "dataset": self.dataset,
                        "llm_called": llm_called,
                        "llm_request_ok": llm_request_ok,
                        "llm_finish_reason": finish_reason,
                        "llm_response_truncated": response_truncated,
                        "initial_json_valid": initial_json_valid,
                        "repair_attempts": attempt,
                        "final_json_valid": True,
                        "candidate_count": candidate_count,
                        "valid_candidate_count": valid_candidate_count,
                        "accepted_candidate_index": accepted_candidate_index,
                        "fallback_used": False,
                        "fallback_reason": None,
                        "config": out[0]["config"],
                    },
                )
                return out
            repair_context = validation_errors[-1] if validation_errors else {"error_type": "schema_error", "message": "no candidates"}
            _append_trace(trace_root, "llm_repair_attempts.jsonl", {"attempt": attempt + 1, "repair_context": repair_context})

        fallback = propose_llm(count=count, history=history)
        for proposal in fallback:
            proposal["proposal_source"] = "fallback"
        fallback_reason = validation_errors[-1] if validation_errors else {"error_type": "no_valid_candidate"}
        _append_trace(trace_root, "llm_fallback_usage.jsonl", {"fallback_used": True, "fallback_reason": fallback_reason})
        if fallback:
            _append_trace(
                trace_root,
                "llm_proposal_trace.jsonl",
                {
                    "proposal_source": "fallback",
                    "trial_index": self.trial_index,
                    "method": self.method,
                    "dataset": self.dataset,
                    "llm_called": llm_called,
                    "llm_request_ok": llm_request_ok,
                    "llm_finish_reason": finish_reason,
                    "llm_response_truncated": response_truncated,
                    "initial_json_valid": initial_json_valid,
                    "repair_attempts": max_repair_retries,
                    "final_json_valid": False,
                    "candidate_count": candidate_count,
                    "valid_candidate_count": 0,
                    "accepted_candidate_index": None,
                    "fallback_used": True,
                    "fallback_reason": fallback_reason,
                    "config": fallback[0]["config"],
                },
            )
        return fallback

    def save_plan(self, path: str | Path, *, count: int = 24) -> None:
        payload = {"agent": "Search Agent", "strategy": self.strategy, "seed": self.seed, "backend": self.backend.public_dict() if self.backend else None, "proposals": self.propose(count=count)}
        write_json(path, payload)

