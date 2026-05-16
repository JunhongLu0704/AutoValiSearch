from __future__ import annotations

from typing import Dict


def build_validation_view_transforms() -> Dict[str, str]:
    return {
        "source_val": "standard",
        "color_jitter_low": "color_jitter_low",
        "color_jitter_medium": "color_jitter_medium",
        "gaussian_blur_low": "gaussian_blur_low",
        "gaussian_blur_medium": "gaussian_blur_medium",
        "grayscale": "grayscale",
        "noise_low": "noise_low",
        "random_resized_crop_mild": "random_resized_crop_mild",
    }

