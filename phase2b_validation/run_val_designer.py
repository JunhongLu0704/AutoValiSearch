from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm.backend_config import load_backend_choice
from llm.json_response import loads_json
from llm.openai_compatible_client import OpenAICompatibleClient
from llm.prompts import validation_designer_messages
from llm.schemas import LLMMessage
from phase2b_validation.baseline_protocols import build_phase2b_baseline_protocols
from phase2b_validation.evaluate_protocols import evaluate_protocols_against_scores
import os

from phase2b_validation.policy_compiler import compile_policy
from phase2b_validation.policy_dsl import PolicyValidationError, policy_schema_summary
from phase2b_validation.policy_evaluator import evaluate_policy
from phase2b_validation.policy_memory import empty_policy_memory, update_val_policy_memory
from phase2b_validation.validation_view_cache import ensure_view_scores_many
from phase2b_validation.validation_view_executor import demo_mapped_view_evaluator_many, real_validation_view_evaluator_many
from phase2b_validation.val_design_memory import summarize_val_design_memory
from phase2b_validation.validator_dsl import compile_protocol
from phase2b_validation.val_strategy_trace import render_val_strategy_trace
from utils.io import append_jsonl, write_csv, write_json, write_jsonl, write_text


def _phase2b_log(message: str) -> None:
    print(f"[Phase II-B] {message}", flush=True)


def build_round0_diagnosis(brief: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round": 0,
        "diagnosis": {
            "main_validation_problem": "source validation can select checkpoints that do not maximize test-side generalization",
            "likely_failure_modes": [
                "late checkpoints may overfit source validation",
                "source-only validation may not capture perturbation robustness",
            ],
            "what_a_good_validator_should_capture": [
                "clean validation performance",
                "moderate photometric robustness",
                "stability across validation views",
            ],
            "design_principles": [
                "keep source_val mandatory",
                "avoid overly harsh aggregation unless needed",
                "prefer moderate perturbations over severe perturbations",
            ],
            "information_boundary": {
                "allowed": list(brief.get("allowed_information_policy", [])),
                "forbidden": list(brief.get("forbidden_information_policy", [])),
            },
        },
    }


def build_policy_initialization_prompt(brief: Mapping[str, Any], memory_summary: Mapping[str, Any], previous_feedback: Mapping[str, Any] | None) -> list[LLMMessage]:
    payload = {
        "task": "design one validation policy for checkpoint selection",
        "validation_design_brief": dict(brief),
        "policy_memory_summary": dict(memory_summary),
        "previous_feedback": dict(previous_feedback or {}),
        "allowed_policy_schema": policy_schema_summary(),
        "parameter_semantics": dict(brief.get("parameter_semantics", {})),
        "augmentation_semantics": dict(brief.get("augmentation_semantics", {})),
        "controller_contract": [
            "The Controller validates your policy JSON.",
            "The Controller builds missing validation views.",
            "The Controller evaluates the policy on 12 epochs x 4 splits x 2 seeds when the checkpoint pool is complete.",
            "The Controller returns policy-level and view-level feedback.",
            "You may use up to 3 new views in one policy when they are complementary.",
        ],
        "you_can_control": [
            "validation augmentation views",
            "view weights",
            "score normalization",
            "aggregation",
            "epoch selection rule",
            "safety fallback rule",
        ],
        "forbidden": [
            "raw checkpoint-level test table",
            "oracle epoch per split/seed",
            "direct epoch labels",
            "any Python code",
        ],
        "augmentation_registry": policy_schema_summary()["allowed_augmentation_operators"],
        "constraints": [
            "Return JSON only.",
            "Use only the listed augmentation operators and legal parameter values.",
            "Do not invent operator names or continuous values outside the grid.",
            "Always include top-level policy_name, new_views, policy, design_hypothesis, and expected_failure_mode.",
            "Treat the parameter and augmentation semantics as the authoritative meaning of each knob.",
        ],
        "output_schema": {
            "policy": {
                "policy_name": "source_dominant_dual_view_v1",
                "new_views": [
                    {
                        "name": "weak_color_stability_v1",
                        "operator": "color_jitter",
                        "params": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02},
                    },
                    {
                        "name": "weak_blur_stability_v1",
                        "operator": "gaussian_blur",
                        "params": {"sigma": 0.75},
                    }
                ],
                "policy": {
                    "views": ["source_val", "weak_color_stability_v1", "weak_blur_stability_v1"],
                    "normalization": "rank_within_split_seed",
                    "aggregation": {"type": "weighted_mean", "weights": {"source_val": 0.85, "weak_color_stability_v1": 0.10, "weak_blur_stability_v1": 0.05}},
                    "epoch_selection": {"type": "argmax_score", "tie_break": "later_epoch"},
                    "safety_rule": {"compare_with_vanilla": True, "fallback_to_vanilla_if_underperform": True},
                },
                "design_hypothesis": "why this policy should help",
                "expected_failure_mode": "how it can fail",
            }
        },
    }
    return [
        LLMMessage(role="system", content="You are designing validation policies for checkpoint selection. Return JSON only."),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def _fallback_policy(dataset: str, run_index: int, round_index: int) -> dict[str, Any]:
    suffix = f"r{run_index:02d}_round{round_index:02d}"
    families = [
        ("weak_color", "color_jitter", {"brightness": 0.10, "contrast": 0.10, "saturation": 0.10, "hue": 0.02}, "rank_within_split_seed", 0.95, {"type": "argmax_score", "tie_break": "later_epoch"}),
        ("blur_low", "gaussian_blur", {"sigma": 0.50}, "zscore_within_split_seed", 0.90, {"type": "argmax_score", "tie_break": "later_epoch"}),
        ("noise_low", "noise", {"std": 0.02}, "minmax_within_split_seed", 0.85, {"type": "latest_within_epsilon_of_best", "epsilon": 0.02}),
        ("crop_mild", "random_resized_crop", {"scale_min": 0.90, "scale_max": 1.0, "ratio_min": 0.95, "ratio_max": 1.05}, "rank_within_split_seed", 0.90, {"type": "earliest_within_epsilon_of_best", "epsilon": 0.01}),
    ]
    family, operator, params, normalization, source_weight, epoch_rule = families[round_index % len(families)]
    view_name = f"{family}_{suffix}"
    return {
        "policy_name": f"{dataset.lower()}_policy_{suffix}",
        "new_views": [
            {
                "name": view_name,
                "operator": operator,
                "params": params,
            }
        ],
        "policy": {
            "views": ["source_val", view_name],
            "normalization": normalization,
            "aggregation": {"type": "weighted_mean", "weights": {"source_val": source_weight, view_name: round(1.0 - source_weight, 6)}},
            "epoch_selection": epoch_rule,
            "safety_rule": {"compare_with_vanilla": True, "fallback_to_vanilla_if_underperform": True},
        },
        "design_hypothesis": "source validation is strong, so the new view is used as weak regularization.",
        "expected_failure_mode": "if perturbation hurts selection, the controller safety rule falls back to vanilla.",
    }


def _propose_policy(
    brief: Mapping[str, Any],
    *,
    backend: Any,
    run_index: int,
    round_index: int,
    memory_summary: Mapping[str, Any],
    previous_feedback: Mapping[str, Any] | None,
    seen_signatures: set[str] | None = None,
    max_tokens: int = 8192,
    repair_max_tokens: int | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = str(brief.get("dataset", "PACS")).upper()
    fallback = _fallback_policy(dataset, run_index, round_index)
    if backend is None:
        policy = _first_non_duplicate_fallback(dataset, run_index, round_index, seen_signatures or set())
        return compile_policy(policy), {"mode": "fallback", "proposal_source": "fallback", "fallback_used": True, "json_valid": True, "repair_attempted": False}
    client = OpenAICompatibleClient(backend, trace_dir=None)
    messages = build_policy_initialization_prompt(brief, memory_summary, previous_feedback)
    start = time.time()
    trace: dict[str, Any] = {"mode": "llm", "fallback_used": False, "json_valid": False, "repair_attempted": False}
    prompt_payload = client.build_payload(messages, temperature=0.2, max_tokens=int(max_tokens))
    _append_llm_trace(output_dir, "llm_raw_prompts", {"run_index": run_index, "round_index": round_index, "prompt": prompt_payload})
    try:
        response = client.chat(messages, temperature=0.2, max_tokens=int(max_tokens))
        finish_reason = response.get("choices", [{}])[0].get("finish_reason") if isinstance(response.get("choices"), list) and response.get("choices") else None
        response_truncated = finish_reason == "length"
        _append_llm_trace(
            output_dir,
            "llm_raw_responses",
            {
                "run_index": run_index,
                "round_index": round_index,
                "response": response,
                "finish_reason": finish_reason,
                "llm_response_truncated": response_truncated,
            },
        )
        parsed = loads_json(response["choices"][0]["message"]["content"])
        policy = compile_policy(_extract_policy_payload(parsed))
        if policy["policy_signature"] in (seen_signatures or set()):
            raise ValueError("duplicate_policy")
        trace.update({
            "response": response,
            "json_valid": True,
            "latency_sec": round(time.time() - start, 4),
            "proposal_source": "llm",
            "max_tokens": int(max_tokens),
            "finish_reason": finish_reason,
            "llm_response_truncated": response_truncated,
        })
        return policy, trace
    except Exception as exc:
        trace["validation_error"] = str(exc)
        trace["repair_attempted"] = True
        _append_llm_trace(
            output_dir,
            "llm_validation_errors",
            {
                "run_index": run_index,
                "round_index": round_index,
                "validation_error": str(exc),
                "max_tokens": int(max_tokens),
                "finish_reason": trace.get("finish_reason"),
                "llm_response_truncated": trace.get("llm_response_truncated", False),
            },
        )
        repair_payload = {
            "task": "repair the invalid validation policy JSON",
            "validation_error": str(exc),
            "allowed_policy_schema": policy_schema_summary(),
            "previous_feedback": dict(previous_feedback or {}),
            "constraints": [
                "Return JSON only.",
                "Do not include forbidden fields.",
                "Do not include Python code.",
                "Keep top-level policy_name, new_views, policy, design_hypothesis, and expected_failure_mode.",
            ],
        }
        try:
            repair_messages = messages + [LLMMessage(role="user", content=json.dumps(repair_payload, ensure_ascii=False, sort_keys=True))]
            repair_prompt_payload = client.build_payload(
                repair_messages,
                temperature=0.1,
                max_tokens=int(repair_max_tokens if repair_max_tokens is not None else max_tokens),
            )
            _append_llm_trace(
                output_dir,
                "llm_repair_prompts",
                {"run_index": run_index, "round_index": round_index, "prompt": repair_prompt_payload},
            )
            response = client.chat(
                repair_messages,
                temperature=0.1,
                max_tokens=int(repair_max_tokens if repair_max_tokens is not None else max_tokens),
            )
            repair_finish_reason = response.get("choices", [{}])[0].get("finish_reason") if isinstance(response.get("choices"), list) and response.get("choices") else None
            repair_response_truncated = repair_finish_reason == "length"
            _append_llm_trace(
                output_dir,
                "llm_repair_responses",
                {
                    "run_index": run_index,
                    "round_index": round_index,
                    "response": response,
                    "finish_reason": repair_finish_reason,
                    "llm_response_truncated": repair_response_truncated,
                },
            )
            parsed = loads_json(response["choices"][0]["message"]["content"])
            policy = compile_policy(_extract_policy_payload(parsed))
            if policy["policy_signature"] in (seen_signatures or set()):
                raise ValueError("duplicate_policy")
            trace.update(
                {
                    "repair_response": response,
                    "json_valid": True,
                    "fallback_used": False,
                    "latency_sec": round(time.time() - start, 4),
                    "proposal_source": "repair",
                    "repair_max_tokens": int(repair_max_tokens if repair_max_tokens is not None else max_tokens),
                    "repair_finish_reason": repair_finish_reason,
                    "repair_llm_response_truncated": repair_response_truncated,
                }
            )
            return policy, trace
        except Exception as repair_exc:
            policy = compile_policy(_first_non_duplicate_fallback(dataset, run_index, round_index, seen_signatures or set()))
            trace.update(
                {
                    "fallback_used": True,
                    "json_valid": False,
                    "repair_error": str(repair_exc),
                    "latency_sec": round(time.time() - start, 4),
                    "proposal_source": "fallback",
                    "repair_max_tokens": int(repair_max_tokens if repair_max_tokens is not None else max_tokens),
                }
            )
            _append_llm_trace(
                output_dir,
                "llm_repair_errors",
                {
                    "run_index": run_index,
                    "round_index": round_index,
                    "validation_error": str(exc),
                    "repair_error": str(repair_exc),
                    "max_tokens": int(max_tokens),
                    "repair_max_tokens": int(repair_max_tokens if repair_max_tokens is not None else max_tokens),
                },
            )
            return policy, trace


def _first_non_duplicate_fallback(dataset: str, run_index: int, round_index: int, seen_signatures: set[str]) -> dict[str, Any]:
    for offset in range(48):
        candidate = _fallback_policy(dataset, run_index, round_index + offset)
        compiled = compile_policy(candidate)
        if compiled["policy_signature"] not in seen_signatures:
            return candidate
    return _fallback_policy(dataset, run_index, round_index)


def _extract_policy_payload(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, Mapping):
        if "policy_name" in parsed:
            return dict(parsed)
        nested_policy = parsed.get("policy")
        if isinstance(nested_policy, Mapping) and "policy_name" in nested_policy:
            return dict(nested_policy)
    raise ValueError("missing required field: policy_name")


def _llm_trace_dir(output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    trace_dir = output_dir / "llm_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def _append_llm_trace(output_dir: Path | None, name: str, payload: Mapping[str, Any]) -> None:
    if output_dir is None:
        return
    (output_dir / "llm_trace").mkdir(parents=True, exist_ok=True)
    append_jsonl(output_dir / "llm_trace" / f"{name}.jsonl", dict(payload))


def build_policy_feedback(metrics: Mapping[str, Any], view_feedback: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    improvement = float(metrics.get("deployable_improvement_over_vanilla", metrics.get("improvement_over_vanilla", 0.0)) or 0.0)
    diagnosis = [
        "The policy improves over vanilla." if improvement >= 0.0 else "The policy underperforms vanilla.",
        "Source validation remains a strong signal.",
        "Use source_val-dominant weights or rank normalization when perturbation views are unstable.",
    ]
    return {
        "policy_feedback": dict(metrics),
        "view_feedback": [
            {
                **dict(item),
                "effect_summary": "cached validation view reused" if item.get("cache_hit") else "validation view evaluated and cached",
            }
            for item in view_feedback
        ],
        "diagnosis": diagnosis,
    }


def render_policy_trace(diagnosis: Mapping[str, Any], rounds: Sequence[Mapping[str, Any]], final_summary: Mapping[str, Any]) -> str:
    lines = ["# Phase II-B Validation Policy Trace", "", "Val Designer uses a controlled validation policy interface. The Controller validates JSON policies, builds cached views, evaluates policy metrics, and enforces safety fallback.", ""]
    lines.append("## Round 0 Diagnosis")
    lines.append(json.dumps(diagnosis, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("")
    for item in rounds:
        policy = item["policy"]
        feedback = item["feedback"]
        lines.append(f"## Run {item['run_idx']} Round {item['round_idx']}")
        lines.append(f"- Policy: {policy['policy_name']}")
        lines.append(f"- Views: {', '.join(policy['policy']['views'])}")
        lines.append(f"- Selected test mean: {feedback['policy_feedback']['selected_checkpoint_test_mean']}")
        lines.append(f"- Improvement over vanilla: {feedback['policy_feedback']['improvement_over_vanilla']}")
        lines.append(f"- Safety fallback used: {feedback['policy_feedback']['used_safety_fallback']}")
        lines.append("")
    lines.append("## Final Summary")
    lines.append(json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return "\n".join(lines)


def run_policy_feedback_loop(
    brief: Mapping[str, Any],
    *,
    backend: Any = None,
    score_rows: Sequence[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    view_cache_dir: str | Path | None = None,
    mode: str = "formal",
) -> dict[str, Any]:
    if score_rows is None:
        raise ValueError("Phase II-B policy mode requires checkpoint score rows")
    dataset = str(brief.get("dataset", "PACS")).upper()
    diagnosis = build_round0_diagnosis(brief)
    output_dir = Path(output_dir) if output_dir is not None else None
    cache_dir = Path(view_cache_dir) if view_cache_dir is not None else (output_dir / "view_cache" if output_dir is not None else Path("view_cache"))
    rounds_payload: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    feedback_rounds: list[dict[str, Any]] = []
    policy_results: list[dict[str, Any]] = []
    memory = empty_policy_memory()
    previous_feedback: dict[str, Any] | None = None
    seen_signatures: set[str] = set()
    total_new_view_signatures: set[str] = set()
    max_new_views_per_round = int(os.environ.get("MAX_NEW_VIEWS_PER_ROUND", "3") or 3)
    max_total_new_views = int(os.environ.get("MAX_TOTAL_NEW_VIEWS_PER_DATASET", "96") or 96)
    designer_max_tokens = int(os.environ.get("VAL_DESIGNER_MAX_TOKENS", "8192") or 8192)
    designer_repair_max_tokens = int(os.environ.get("VAL_DESIGNER_REPAIR_MAX_TOKENS", str(designer_max_tokens)) or designer_max_tokens)
    num_runs = int(brief.get("num_runs", 1))
    rounds = int(brief.get("rounds", 24))
    evaluator = demo_mapped_view_evaluator_many if str(mode).lower() == "demo" else real_validation_view_evaluator_many
    if output_dir is not None:
        trace_path = output_dir / "policy_search_trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        _llm_trace_dir(output_dir)

    _phase2b_log(
        f"policy loop start dataset={dataset} runs={num_runs} rounds={rounds} backend={'yes' if backend else 'no'} score_rows={len(score_rows)}"
    )

    for run_index in range(num_runs):
        for round_index in range(rounds):
            if output_dir is not None:
                append_jsonl(
                    output_dir / "phase2b_val_designer_trace.jsonl",
                    {
                        "dataset": dataset,
                        "event": "proposal_start",
                        "run_idx": run_index,
                        "round_idx": round_index,
                        "backend": backend.public_dict() if backend else None,
                    },
                )
            _phase2b_log(
                f"policy round start dataset={dataset} run={run_index + 1}/{num_runs} round={round_index + 1}/{rounds}"
            )
            policy, trace = _propose_policy(
                brief,
                backend=backend,
                run_index=run_index,
                round_index=round_index,
                memory_summary=memory,
                previous_feedback=previous_feedback,
                seen_signatures=seen_signatures,
                max_tokens=designer_max_tokens,
                repair_max_tokens=designer_repair_max_tokens,
                output_dir=output_dir,
            )
            duplicate_policy = policy["policy_signature"] in seen_signatures
            if duplicate_policy:
                policy = compile_policy(_first_non_duplicate_fallback(dataset, run_index, round_index, seen_signatures))
                trace = {**trace, "proposal_source": "fallback", "fallback_used": True, "duplicate_policy": True}
            if len(policy.get("required_views", [])) > max_new_views_per_round or len(total_new_view_signatures | {view["signature"] for view in policy.get("required_views", [])}) > max_total_new_views:
                policy = compile_policy(_first_non_duplicate_fallback(dataset, run_index, round_index + 7, seen_signatures))
                trace = {**trace, "proposal_source": "fallback", "fallback_used": True, "view_budget_exceeded": True}
            seen_signatures.add(policy["policy_signature"])
            total_new_view_signatures |= {view["signature"] for view in policy.get("required_views", [])}
            view_rows, view_feedback = ensure_view_scores_many(
                dataset=dataset,
                score_rows=score_rows,
                views=policy["required_views"],
                cache_root=cache_dir,
                evaluator=evaluator,
            )
            _phase2b_log(
                f"policy round views done dataset={dataset} run={run_index + 1}/{num_runs} round={round_index + 1}/{rounds} views={len(policy['required_views'])} cache_hits={sum(1 for item in view_feedback if item.get('cache_hit'))}/{len(view_feedback)}"
            )
            metrics, selected_rows = evaluate_policy(policy, score_rows, view_rows, dataset=dataset, run_idx=run_index, round_idx=round_index)
            feedback = build_policy_feedback(metrics, view_feedback)
            memory = update_val_policy_memory(memory, policy, feedback)
            previous_feedback = feedback
            round_record = {
                "run_idx": run_index,
                "round_idx": round_index,
                "policy": policy,
                "feedback": feedback,
                "trace": trace,
                "selected_rows": selected_rows,
            }
            rounds_payload.append(round_record)
            policies.append(policy)
            feedback_rounds.append(feedback)
            policy_results.append(dict(metrics))
            if output_dir is not None:
                append_jsonl(
                    output_dir / "phase2b_val_designer_trace.jsonl",
                    {
                        "dataset": dataset,
                        "event": "proposal_end",
                        "run_idx": run_index,
                        "round_idx": round_index,
                        "proposal_source": trace.get("proposal_source") or ("fallback" if trace.get("fallback_used") else "llm"),
                        "fallback_used": bool(trace.get("fallback_used", False)),
                    },
                )
                run_dir = output_dir / "policy_search" / f"run_{run_index}"
                write_json(run_dir / f"round_{round_index:03d}_policy.json", policy)
                write_json(run_dir / f"round_{round_index:03d}_feedback.json", feedback)
                append_jsonl(
                    output_dir / "policy_search_trace.jsonl",
                    {
                        "dataset": dataset,
                        "run_idx": run_index,
                        "round_idx": round_index,
                        "proposal_source": trace.get("proposal_source") or ("fallback" if trace.get("fallback_used") else "llm"),
                        "policy_name": policy["policy_name"],
                        "policy_signature": policy["policy_signature"],
                        "new_view_signatures": [view["signature"] for view in policy.get("required_views", [])],
                        "duplicate_policy": bool(trace.get("duplicate_policy", duplicate_policy)),
                        "repair_attempts": 1 if trace.get("repair_attempted") else 0,
                        "max_tokens": trace.get("max_tokens"),
                        "repair_max_tokens": trace.get("repair_max_tokens"),
                        "finish_reason": trace.get("finish_reason"),
                        "repair_finish_reason": trace.get("repair_finish_reason"),
                        "llm_response_truncated": trace.get("llm_response_truncated"),
                        "repair_llm_response_truncated": trace.get("repair_llm_response_truncated"),
                        "validation_error": trace.get("validation_error"),
                        "repair_error": trace.get("repair_error"),
                        "view_cache_hits": [item["view_signature"] for item in view_feedback if item.get("cache_hit")],
                        "view_cache_misses": [item["view_signature"] for item in view_feedback if not item.get("cache_hit")],
                        "selected_checkpoint_test_mean": metrics.get("selected_checkpoint_test_mean"),
                        "deployable_selected_test_mean": metrics.get("deployable_selected_test_mean"),
                        "improvement_over_vanilla": metrics.get("improvement_over_vanilla"),
                        "deployable_improvement_over_vanilla": metrics.get("deployable_improvement_over_vanilla"),
                        "used_safety_fallback": metrics.get("used_safety_fallback"),
                        "memory_recommendation": memory.get("recommended_next_changes", []),
                    },
                )
            _phase2b_log(
                f"policy round done dataset={dataset} run={run_index + 1}/{num_runs} round={round_index + 1}/{rounds} source={trace.get('proposal_source') or ('fallback' if trace.get('fallback_used') else 'llm')} selected_mean={metrics.get('selected_checkpoint_test_mean')} deployable_mean={metrics.get('deployable_selected_test_mean')} safety_fallback={metrics.get('used_safety_fallback')}"
            )

    best_policy = max(policy_results, key=lambda row: float(row.get("selected_checkpoint_test_mean", 0.0) or 0.0), default={})
    best_deployable_policy = max(policy_results, key=lambda row: float(row.get("deployable_selected_test_mean", row.get("selected_checkpoint_test_mean", 0.0)) or 0.0), default={})
    final_summary = {
        "dataset": dataset,
        "best_policy": best_policy,
        "best_deployable_policy": best_deployable_policy,
        "memory_summary": memory,
        "policy_count": len(policy_results),
    }
    trace_markdown = render_policy_trace(diagnosis, rounds_payload, final_summary)
    return {
        "diagnosis": diagnosis,
        "policies": policies,
        "policy_rounds": rounds_payload,
        "feedback_rounds": feedback_rounds,
        "policy_results": policy_results,
        "memory_summary": memory,
        "final_summary": final_summary,
        "trace_markdown": trace_markdown,
    }


def _history_protocol_names(history: Sequence[Mapping[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for item in history or []:
        name = str(item.get("protocol_name") or "").strip()
        if name:
            names.add(name)
    return names


def _coerce_protocol(raw_protocol: Mapping[str, Any], *, dataset: str, run_index: int, round_index: int) -> dict[str, Any]:
    protocol = dict(raw_protocol)
    protocol["protocol_name"] = str(protocol.get("protocol_name") or f"{dataset.lower()}_llm_validator_r{run_index:02d}_round{round_index:02d}")
    protocol["design_hypothesis"] = str(protocol.get("design_hypothesis") or "history-conditioned validation design")
    protocol["expected_advantage"] = str(protocol.get("expected_advantage") or "better checkpoint selection under mild shift")
    protocol["risk"] = str(protocol.get("risk") or "may be too conservative")
    protocol["selection_rule"] = str(protocol.get("selection_rule") or "select_epoch_with_max_protocol_score")
    return compile_protocol(protocol)


def _design_protocol(
    brief: Mapping[str, Any],
    *,
    backend: Any = None,
    run_index: int,
    round_index: int,
    memory_summary: Mapping[str, Any],
    previous_feedback: Mapping[str, Any] | None,
    history: Sequence[Mapping[str, Any]] | None = None,
    diagnosis: Mapping[str, Any] | None = None,
    max_tokens: int = 8192,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = str(brief.get("dataset", "PACS")).upper()
    fallback = {
        "protocol_name": f"{dataset.lower()}_llm_validator_r{run_index:02d}_round{round_index:02d}",
        "views": ["source_val", "color_jitter_low", "gaussian_blur_low"],
        "aggregation": "mean_minus_std",
        "alpha": 0.1,
        "selection_rule": "select_epoch_with_max_protocol_score",
        "design_hypothesis": "moderate photometric robustness improves checkpoint selection",
        "expected_advantage": "keeps source_val while discouraging brittle checkpoints",
        "risk": "can underweight very strong clean checkpoints",
    }
    if backend is None:
        return _coerce_protocol(fallback, dataset=dataset, run_index=run_index, round_index=round_index), {"mode": "fallback"}

    client = OpenAICompatibleClient(backend, trace_dir=None)
    messages = validation_designer_messages(
        brief,
        int(brief.get("num_runs", 1)),
        int(brief.get("rounds", 24)),
        history=history,
        run_index=run_index,
        round_index=round_index,
        memory_summary=memory_summary,
        previous_round_feedback=previous_feedback,
        diagnosis=diagnosis,
    )
    payload = client.build_payload(messages, temperature=0.2, max_tokens=int(max_tokens))
    start = time.time()
    try:
        response = client.chat(messages, temperature=0.2, max_tokens=int(max_tokens))
        content = response["choices"][0]["message"]["content"]
        parsed = loads_json(content)
        raw_protocol = parsed.get("protocol")
        if raw_protocol is None and isinstance(parsed.get("protocols"), list) and parsed["protocols"]:
            raw_protocol = parsed["protocols"][0]
        if not isinstance(raw_protocol, Mapping):
            raise ValueError("LLM response missing protocol object")
        protocol = _coerce_protocol(raw_protocol, dataset=dataset, run_index=run_index, round_index=round_index)
        return protocol, {
            "mode": "llm",
            "prompt": payload,
            "response": response,
            "json_valid": True,
            "fallback_used": False,
            "latency_sec": round(time.time() - start, 4),
            "error": None,
        }
    except Exception as exc:
        protocol = _coerce_protocol(fallback, dataset=dataset, run_index=run_index, round_index=round_index)
        return protocol, {
            "mode": "fallback",
            "prompt": payload,
            "response": None,
            "json_valid": False,
            "fallback_used": True,
            "latency_sec": round(time.time() - start, 4),
            "error": str(exc),
        }


def _select_baseline_rows(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> dict[str, dict[str, Any]]:
    baseline_protocols = build_phase2b_baseline_protocols(dataset)
    result_rows, _, _ = evaluate_protocols_against_scores(baseline_protocols, score_rows, dataset=dataset)
    return {row["protocol_name"]: row for row in result_rows}


def _demo_validation_scores(brief: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = str(brief.get("dataset", "PACS")).upper()
    rows: list[dict[str, Any]] = []
    for split_index, split_name in enumerate(["demo_split_0", "demo_split_1"], start=1):
        for seed in [0, 1]:
            for epoch in [1, 2]:
                base = 68.0 + split_index * 1.5 + seed * 0.4 + epoch * 0.6
                row = {
                    "checkpoint_id": f"{dataset.lower()}_{split_name}_seed{seed}_epoch{epoch}",
                    "dataset": dataset,
                    "split": split_name,
                    "seed": seed,
                    "epoch": epoch,
                    "checkpoint_path": f"<demo>/{dataset.lower()}/{split_name}/seed{seed}/epoch_{epoch:03d}.pth",
                    "status": "ok",
                    "fail_reason": None,
                    "error_message": None,
                    "selection_anchor": round(base - 1.1, 6),
                    "source_val": round(base, 6),
                    "color_jitter_low": round(base + 0.3, 6),
                    "color_jitter_medium": round(base + 0.15, 6),
                    "gaussian_blur_low": round(base + 0.18, 6),
                    "gaussian_blur_medium": round(base - 0.05, 6),
                    "grayscale": round(base - 0.02, 6),
                    "noise_low": round(base + 0.08, 6),
                    "random_resized_crop_mild": round(base + 0.12, 6),
                    "test_acc": round(base - 0.9, 6),
                }
                rows.append(row)
    return rows


def design_protocol_round(
    brief: Mapping[str, Any],
    *,
    run_index: int,
    round_index: int,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    protocol, _ = _design_protocol(
        brief,
        backend=None,
        run_index=run_index,
        round_index=round_index,
        memory_summary={},
        previous_feedback=None,
        history=history,
        diagnosis=build_round0_diagnosis(brief),
    )
    return protocol


def design_protocol_rounds(
    brief: Mapping[str, Any],
    *,
    num_runs: int = 1,
    rounds: int = 24,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    protocols: list[dict[str, Any]] = []
    external_history = list(history or [])
    for run_index in range(num_runs):
        run_history: list[dict[str, Any]] = []
        for round_index in range(rounds):
            protocol = design_protocol_round(
                brief,
                run_index=run_index,
                round_index=round_index,
                history=external_history + run_history,
            )
            protocols.append(protocol)
            run_history.append(protocol)
    return protocols


def run_feedback_loop(
    brief: Mapping[str, Any],
    *,
    backend: Any = None,
    score_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    dataset = str(brief.get("dataset", "PACS")).upper()
    if score_rows is None:
        raise ValueError("Phase II-B formal mode requires checkpoint score rows")
    diagnosis = build_round0_diagnosis(brief)
    protocol_rounds: list[dict[str, Any]] = []
    feedback_rounds: list[dict[str, Any]] = []
    num_runs = int(brief.get("num_runs", 1))
    rounds = int(brief.get("rounds", 24))
    baseline_rows = _select_baseline_rows(score_rows, dataset=dataset)
    vanilla_row = baseline_rows.get(f"{dataset.lower()}_vanilla_best_val", {})
    best_test_upper_bound_row = baseline_rows.get(f"{dataset.lower()}_best_test_upper_bound", {})
    final_epoch_row = baseline_rows.get(f"{dataset.lower()}_final_epoch", {})
    designer_max_tokens = int(os.environ.get("VAL_DESIGNER_MAX_TOKENS", "8192") or 8192)

    for run_index in range(num_runs):
        run_protocol_history: list[dict[str, Any]] = []
        run_protocol_rounds: list[dict[str, Any]] = []
        run_feedback_rounds: list[dict[str, Any]] = []
        memory_summary = summarize_val_design_memory(run_protocol_rounds, run_feedback_rounds)
        previous_feedback: dict[str, Any] | None = None
        for round_index in range(rounds):
            global_round = run_index * rounds + round_index + 1
            protocol, trace_payload = _design_protocol(
                brief,
                backend=backend,
                run_index=run_index,
                round_index=round_index,
                memory_summary=memory_summary,
                previous_feedback=previous_feedback,
                history=protocol_rounds + run_protocol_history,
                diagnosis=diagnosis,
                max_tokens=designer_max_tokens,
            )
            if trace_payload.get("fallback_used"):
                trace_payload["validation_error"] = trace_payload.get("error")
            protocol_rounds.append(
                {
                    "run": run_index,
                    "round": global_round,
                    "round_in_run": round_index + 1,
                    "protocols": [protocol],
                    "trace": trace_payload,
                }
            )
            run_protocol_rounds.append(protocol_rounds[-1])
            run_protocol_history.append(protocol)
            memory_summary = summarize_val_design_memory(run_protocol_rounds, run_feedback_rounds)
            protocol_results, _, random_summary = evaluate_protocols_against_scores([protocol], score_rows, dataset=dataset)
            protocol_result = protocol_results[0]
            previous_feedback = {
                "round": global_round,
                "run": run_index,
                "round_in_run": round_index + 1,
                "round_result_summary": {
                    "best_protocol": protocol["protocol_name"],
                    "best_selected_mean_test_acc": protocol_result["selected_checkpoint_test_mean"],
                    "best_selection_regret": protocol_result["selection_regret"],
                    "vanilla_selected_mean_test_acc": float(vanilla_row.get("selected_checkpoint_test_mean", 0.0) or 0.0),
                    "best_test_upper_bound_mean_test_acc": float(best_test_upper_bound_row.get("selected_checkpoint_test_mean", 0.0) or 0.0),
                    "final_epoch_selected_mean_test_acc": float(final_epoch_row.get("selected_checkpoint_test_mean", 0.0) or 0.0),
                    "best_improvement_over_vanilla": round(
                        float(protocol_result["selected_checkpoint_test_mean"]) - float(vanilla_row.get("selected_checkpoint_test_mean", 0.0) or 0.0),
                        6,
                    ),
                },
                "protocol_results": protocol_results,
                "random_summary": random_summary,
                "memory_summary": memory_summary,
                "observed_patterns": [
                    "source_val plus one or two moderate views is usually safer than severe perturbations",
                    "mean_minus_std and harmonic_mean remain the main stable aggregations",
                ],
            }
            feedback_rounds.append(previous_feedback)
            run_feedback_rounds.append(previous_feedback)
            memory_summary = summarize_val_design_memory(run_protocol_rounds, run_feedback_rounds)

    final_memory_summary = summarize_val_design_memory(protocol_rounds, feedback_rounds)
    best = final_memory_summary.get("best_protocol_so_far") or {}
    final_summary = {
        "dataset": dataset,
        "best_protocol": best,
        "memory_summary": final_memory_summary,
        "baseline_summary": {
            "final_epoch": final_epoch_row,
            "vanilla_best_val": vanilla_row,
            "best_test_upper_bound": best_test_upper_bound_row,
        },
        "remaining_uncertainty": "whether the same moderate photometric mix transfers equally across datasets",
    }
    return {
        "diagnosis": diagnosis,
        "protocol_rounds": protocol_rounds,
        "feedback_rounds": feedback_rounds,
        "final_summary": final_summary,
        "trace_markdown": render_val_strategy_trace(diagnosis, protocol_rounds, feedback_rounds, final_summary),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Design validation protocols")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--brief-path", required=True)
    parser.add_argument("--validation-scores")
    parser.add_argument("--scores-path", dest="validation_scores", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--protocols-per-round", type=int, default=1)
    parser.add_argument("--backend-config")
    parser.add_argument("--backend-name")
    parser.add_argument("--backend", choices=["cloud", "local_openai_compatible"])
    parser.add_argument("--mode", choices=["formal", "demo"], default="formal")
    parser.add_argument("--use-example-artifacts", action="store_true", help="Compatibility alias for --mode demo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode = "demo" if args.use_example_artifacts else str(args.mode).lower()
    brief = json.loads(Path(args.brief_path).read_text(encoding="utf-8"))
    backend_name = args.backend_name or args.backend
    backend = load_backend_choice(args.backend_config, backend_name) if args.backend_config else None
    brief["num_runs"] = int(args.num_runs)
    brief["rounds"] = int(args.rounds)
    brief["protocols_per_round"] = int(args.protocols_per_round)
    brief["backend_name"] = backend.name if backend else (backend_name or "deterministic_fallback")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]]
    if args.validation_scores:
        score_path = Path(args.validation_scores)
        if score_path.suffix.lower() == ".jsonl":
            score_rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            import csv

            with score_path.open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
    elif mode == "demo":
        score_rows = _demo_validation_scores(brief)
    else:
        raise ValueError("Phase II-B formal mode requires --validation-scores or --scores-path")

    loop = run_policy_feedback_loop(brief, backend=backend, score_rows=score_rows, output_dir=output_dir, view_cache_dir=output_dir / "view_cache", mode=mode)
    diagnosis = loop["diagnosis"]
    policy_rounds = loop["policy_rounds"]
    feedback_rounds = loop["feedback_rounds"]
    policies = list(loop["policies"])
    write_json(output_dir / "val_design_brief.json", brief)
    write_json(output_dir / "round0_diagnosis.json", diagnosis)
    for round_payload in policy_rounds:
        flat_index = int(round_payload["run_idx"]) * int(args.rounds) + int(round_payload["round_idx"])
        write_json(output_dir / f"round{flat_index}_policy.json", round_payload["policy"])
        write_json(output_dir / f"round{flat_index}_feedback.json", round_payload["feedback"])
    write_json(output_dir / "final_policy_summary.json", loop["final_summary"])
    write_text(output_dir / "val_strategy_trace.md", loop["trace_markdown"])
    write_json(output_dir / "phase2b_all_policies.json", policies)
    write_csv(
        output_dir / "policy_results.csv",
        loop["policy_results"],
        [
            "policy_name",
            "run_idx",
            "round_idx",
            "selected_checkpoint_test_mean",
            "deployable_selected_test_mean",
            "vanilla_best_val",
            "improvement_over_vanilla",
            "deployable_improvement_over_vanilla",
            "selection_regret",
            "top3_epoch_hit_rate",
            "top5_epoch_hit_rate",
            "gap_to_best_test_upper_bound",
            "used_safety_fallback",
        ],
    )
    write_json(
        output_dir / "phase2b_val_designer_summary.json",
        {
            "dataset": str(args.dataset).upper(),
            "policy_count": len(policies),
            "rounds": int(args.rounds),
            "runs": int(args.num_runs),
            "protocols_per_round": int(args.protocols_per_round),
            "mode": mode,
            "backend": backend.public_dict() if backend else None,
        },
    )
    trace_rows = [
        {
            "round_index": 0,
            "event": "diagnosis",
            "backend_name": backend.name if backend else (args.backend or "deterministic_fallback"),
            "model": backend.model if backend else "offline-replay",
            "base_url_redacted": backend.base_url if backend else None,
            "latency_sec": 0.0,
            "json_valid": True,
            "fallback_used": backend is None,
        }
    ]
    for round_payload in policy_rounds:
        trace_rows.append(
            {
                "round_index": int(round_payload["run_idx"]) * int(args.rounds) + int(round_payload["round_idx"]),
                "event": "policy",
                "policy_name": round_payload["policy"].get("policy_name"),
                "fallback_used": bool(round_payload.get("trace", {}).get("fallback_used", False)),
                "json_valid": bool(round_payload.get("trace", {}).get("json_valid", True)),
            }
        )
    write_jsonl(output_dir / "phase2b_val_designer_trace.jsonl", trace_rows)
    _phase2b_log(
        f"val designer complete dataset={str(args.dataset).upper()} policies={len(policies)} output_dir={output_dir}"
    )
    print(json.dumps({"dataset": str(args.dataset).upper(), "policy_count": len(policies)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
