from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .schemas import LLMMessage


def _phase1_code_context() -> dict[str, Any]:
    return {
        "normalize_config": [
            "required fields: dataset, split_dir, image_root, bn_mode, budget, bs, lr, lambdap, epochp, num_f",
            "pretrained must be True; validator protocol/spec are normalized here",
            "config['swa_start_epoch'] = config['epochp']",
            "config['stable_after_epochp'] = config['stable_enabled']",
        ],
        "training_switch": [
            "if args.stable_enabled and epoch >= args.epochp and cfeatures.size(0) > 16: sample_weight = stable_weight_learner(...)",
            "stable_weight_learner runs in float32 and applies softmax over weight * all_rsw_weight",
        ],
        "stable_weight_learner": [
            "normalized_weight = F.softmax(weight * all_rsw_weight, dim=0)",
            "_ensure_finite('stable_weights', normalized_weight) happens before lossg.backward()",
        ],
    }


def _phase2b_code_context() -> dict[str, Any]:
    return {
        "policy_schema_summary": [
            "required fields: policy_name, new_views, policy.views, policy.normalization, policy.aggregation, policy.epoch_selection, policy.safety_rule, design_hypothesis, expected_failure_mode",
            "allowed augmentation operators and legal parameter grids are enumerated here",
            "allowed normalizations and aggregations are enumerated here",
        ],
        "ensure_view_scores_many": [
            "groups are formed by split/seed; missing groups are evaluated and cached",
            "VAL_EVAL_PARALLEL controls group-level parallelism via ThreadPoolExecutor",
            "each view gets its own cache signature and merged_scores.csv",
        ],
        "evaluate_checkpoint_group_views_scores": [
            "loads each checkpoint once per checkpoint-path loop",
            "builds a DataLoader for each view and calls evaluate_model(view_loader, model, runtime_args, ...)",
            "checkpoint_bundle is loaded before looping over views",
        ],
    }


def search_agent_messages(
    search_space: Mapping[str, Sequence[Any]],
    count: int,
    history: Sequence[Mapping[str, Any]] | None = None,
    *,
    memory_summary: Mapping[str, Any] | None = None,
    candidates_per_call: int | None = None,
    repair_context: Mapping[str, Any] | None = None,
) -> list[LLMMessage]:
    candidate_count = int(candidates_per_call or max(count, 1))
    payload = {
        "task": "propose candidate bounded training configurations for the next aggregate trial",
        "count": int(count),
        "candidates_per_call": candidate_count,
        "task_context": "Phase I fixed-budget search over stable_swa training hyperparameters.",
        "parameter_semantics": {
            "lr": "main optimizer learning rate for the base classifier and backbone updates; larger values adapt faster but are more likely to destabilize training or trigger NaNs, while smaller values are safer but may underfit within the fixed epoch budget.",
            "lambdap": "weight on the stable-learning / reweighting objective; larger values increase the influence of the auxiliary stability term relative to the base classification loss, while smaller values reduce its effect.",
            "epochp": "warmup length before stable-learning begins; smaller values switch to the stability branch earlier, while larger values keep more pure classification pretraining before the auxiliary objective is activated.",
            "num_f": "number of random Fourier features used to estimate the stability term; larger values increase approximation capacity and compute cost, while smaller values are cheaper but may be noisier.",
        },
        "code_context": _phase1_code_context(),
        "search_space": {key: list(values) for key, values in search_space.items()},
        "search_memory_summary": dict(memory_summary or {}),
        "repair_context": dict(repair_context or {}),
        "recent_trials": list(history or [])[-3:],
        "output_schema": {
            "candidates": [
                {
                    "proposal": {"lr": 0.0, "lambdap": 0.0, "epochp": 0, "num_f": 0},
                    "design_hypothesis": "string",
                    "relation_to_memory": "string",
                    "risk_note": "string",
                }
            ]
        },
        "constraints": [
            "Return JSON only.",
            "Use only values from the search space.",
            "Produce exactly candidates_per_call candidates.",
            "Avoid repeating previously proposed configs when history is present.",
            "Keep proposals diverse.",
            "Use search_memory_summary first; recent_trials are examples only.",
        ],
    }
    return [
        LLMMessage(role="system", content="You are the Search Agent for AutoValiSearch. Output JSON only."),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def validation_designer_messages(
    brief: Mapping[str, Any],
    num_runs: int,
    rounds: int,
    history: Sequence[Mapping[str, Any]] | None = None,
    *,
    run_index: int | None = None,
    round_index: int | None = None,
    memory_summary: Mapping[str, Any] | None = None,
    previous_round_feedback: Mapping[str, Any] | None = None,
    diagnosis: Mapping[str, Any] | None = None,
) -> list[LLMMessage]:
    payload = {
        "task": "design the next validation protocol",
        "validation_design_brief": brief,
        "val_design_memory_summary": dict(memory_summary or {}),
        "previous_round_feedback": dict(previous_round_feedback or {}),
        "round0_diagnosis": dict(diagnosis or {}),
        "code_context": _phase2b_code_context(),
        "schedule": {
            "num_runs": int(num_runs),
            "rounds": int(rounds),
            "current_run_index": run_index,
            "current_round_index": round_index,
        },
        "recent_protocol_examples": list(history or [])[-3:],
        "augmentation_semantics": {
            "source_val": "identity validation on the original source-validation split; this is the mandatory anchor view.",
            "color_jitter_low": "mild photometric perturbation; brightness/contrast/saturation/hue are all adjusted only slightly.",
            "color_jitter_medium": "stronger photometric perturbation than the low setting; useful for testing color robustness.",
            "gaussian_blur_low": "light blur that smooths local details without destroying structure.",
            "gaussian_blur_medium": "stronger blur that removes more high-frequency detail and is a harsher robustness check.",
            "grayscale": "convert to grayscale; removes color cues completely.",
            "noise_low": "add small Gaussian noise to pixels; a gentle corruption robustness test.",
            "random_resized_crop_mild": "mild crop-and-resize augmentation; scale and aspect-ratio ranges are narrow to avoid destructive cropping.",
            "autocontrast": "stretch image contrast dynamically; often changes global intensity distribution without adding geometric distortion.",
            "sharpness_low": "slightly sharpen or soften the image via the sharpness factor; values below 1.0 usually blur relative to the input.",
            "posterize_mid": "reduce color bit depth; this coarsens the image and removes fine color detail.",
            "solarize_mid": "invert pixels above a threshold; this can expose sensitivity to high-intensity regions.",
        },
        "augmentation_parameter_semantics": {
            "brightness": "photometric strength for brightness shift; larger values mean stronger brightness changes.",
            "contrast": "photometric strength for contrast shift; larger values mean stronger contrast changes.",
            "saturation": "photometric strength for saturation shift; larger values mean stronger color saturation changes.",
            "hue": "photometric strength for hue shift; larger absolute values mean stronger hue rotation.",
            "sigma": "blur radius / standard deviation; larger values produce stronger blurring.",
            "p": "application probability or full-strength application depending on the operator; in this repo grayscale uses p=1.0 for full conversion.",
            "std": "noise standard deviation; larger values inject stronger pixel noise.",
            "scale_min": "lower bound on random crop area scale; smaller values allow more aggressive cropping.",
            "scale_max": "upper bound on random crop area scale.",
            "ratio_min": "lower bound on random crop aspect ratio; smaller values allow taller / narrower crops.",
            "ratio_max": "upper bound on random crop aspect ratio; larger values allow wider crops.",
            "factor": "sharpness multiplier; values below 1.0 usually soften the image, values above 1.0 sharpen it.",
            "bits": "posterize bit depth; smaller values remove more color precision and are more aggressive.",
            "threshold": "solarize cutoff; pixels above the threshold are inverted, so lower thresholds are more aggressive.",
        },
        "allowed_views": {
            "source_val": {"operator": "identity", "params": {}},
            "color_jitter_low": {"operator": "color_jitter", "params": {"brightness": 0.10, "contrast": 0.10, "saturation": 0.10, "hue": 0.02}},
            "color_jitter_medium": {"operator": "color_jitter", "params": {"brightness": 0.20, "contrast": 0.20, "saturation": 0.20, "hue": 0.05}},
            "gaussian_blur_low": {"operator": "gaussian_blur", "params": {"sigma": 0.50}},
            "gaussian_blur_medium": {"operator": "gaussian_blur", "params": {"sigma": 1.00}},
            "grayscale": {"operator": "grayscale", "params": {"p": 1.0}},
            "noise_low": {"operator": "noise", "params": {"std": 0.02}},
            "random_resized_crop_mild": {"operator": "random_resized_crop", "params": {"scale_min": 0.90, "scale_max": 1.0, "ratio_min": 0.95, "ratio_max": 1.05}},
            "autocontrast": {"operator": "autocontrast", "params": {}},
            "sharpness_low": {"operator": "sharpness", "params": {"factor": 0.75}},
            "posterize_mid": {"operator": "posterize", "params": {"bits": 6}},
            "solarize_mid": {"operator": "solarize", "params": {"threshold": 128}},
        },
        "allowed_aggregations": ["mean", "harmonic_mean", "mean_minus_std", "weighted_mean", "median", "max", "logsumexp"],
        "output_schema": {
            "protocol": {
                "protocol_name": "string",
                "views": ["source_val", "grayscale"],
                "aggregation": "mean",
                "alpha": 0.0,
                "weights": [1.0, 1.0],
            }
        },
        "constraints": [
            "Return JSON only.",
            "Design exactly one protocol for this round.",
            "Every protocol must include source_val.",
            "Keep protocol names dataset-specific.",
            "Use prior-round history to refine the next round.",
            "Never request or rely on checkpoint-level test tables or per-epoch test curves.",
            "Use only the listed augmentation operators and legal parameter values.",
            "Do not invent operator names or continuous values outside the grid.",
            "You may combine up to 3 validation views when it helps stabilize selection.",
            "Treat the augmentation and parameter semantics as the ground truth for what each view does.",
        ],
    }
    return [
        LLMMessage(role="system", content="You are the Val Designer Agent for AutoValiSearch. Output JSON only."),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def report_agent_messages(evidence_pack: Mapping[str, Any]) -> list[LLMMessage]:
    payload = {
        "task": "generate evidence-based demo report outputs",
        "evidence_pack": dict(evidence_pack),
        "output_schema": {
            "demo_report": "markdown string",
            "readme_summary": "markdown string",
            "resume_snippet": "markdown string",
            "ppt_outline": "markdown string",
        },
        "constraints": [
            "Return JSON only.",
            "Do not invent numbers.",
            "Use only the evidence pack.",
            "Keep the report concise and presentation-ready.",
        ],
    }
    return [
        LLMMessage(role="system", content="You are the Report Agent for AutoValiSearch. Output JSON only."),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]
