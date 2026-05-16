from __future__ import annotations

from typing import Any, Mapping


def build_validation_brief(*, dataset: str, summary: Mapping[str, Any], checkpoint_count: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "checkpoint_count": int(checkpoint_count),
        "summary": dict(summary),
        "parameter_semantics": {
            "lr": "main optimizer learning rate in Phase I; larger values learn faster but can destabilize training, while smaller values are safer but may underfit within the fixed epoch budget.",
            "lambdap": "weight on the stable-learning / reweighting term in Phase I; larger values make the auxiliary stability objective more dominant.",
            "epochp": "warmup length before the stable-learning branch starts in Phase I; smaller values activate the auxiliary objective earlier.",
            "num_f": "number of random Fourier features used by the stability estimator in Phase I; larger values increase approximation capacity and cost.",
        },
        "augmentation_semantics": {
            "source_val": "mandatory identity validation view on the source-validation split.",
            "color_jitter_low": "mild brightness / contrast / saturation / hue perturbation.",
            "color_jitter_medium": "stronger photometric perturbation than the low setting.",
            "gaussian_blur_low": "light blur that removes some high-frequency detail.",
            "gaussian_blur_medium": "stronger blur that removes more local detail.",
            "grayscale": "convert to grayscale and remove color cues.",
            "noise_low": "small pixel-level Gaussian noise.",
            "random_resized_crop_mild": "mild crop-and-resize augmentation with narrow scale and aspect-ratio ranges.",
            "autocontrast": "stretch the intensity range without geometric distortion.",
            "sharpness_low": "slight sharpness adjustment; values below 1.0 soften the image.",
            "posterize_mid": "reduce color bit depth to coarsen the image.",
            "solarize_mid": "invert pixels above a threshold to stress intensity sensitivity.",
        },
        "allowed_information_policy": [
            "protocol-level selected mean test accuracy",
            "protocol-level selection regret",
            "Top-3 hit rate",
            "improvement over vanilla",
            "observed protocol-level patterns",
        ],
        "forbidden_information_policy": [
            "full checkpoint-level test table",
            "per-epoch test curves",
            "oracle epoch per split/seed",
            "direct epoch-selection labels",
        ],
        "allowed_dsl": {
            "views": [
                "source_val",
                "color_jitter_low",
                "color_jitter_medium",
                "gaussian_blur_low",
                "gaussian_blur_medium",
                "grayscale",
                "noise_low",
                "random_resized_crop_mild",
                "autocontrast",
                "sharpness",
                "posterize",
                "solarize",
            ],
            "aggregations": ["mean", "harmonic_mean", "mean_minus_std", "weighted_mean", "median", "max", "logsumexp"],
            "selection_rule": "select_epoch_with_max_protocol_score",
        },
        "diagnosis": [
            "vanilla best-val may not align with test.",
            "use source_val plus robustness views to reduce regret.",
        ],
    }
