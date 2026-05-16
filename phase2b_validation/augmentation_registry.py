from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class AugmentationError(ValueError):
    pass


@dataclass(frozen=True)
class AugmentationSpec:
    operator: str
    grid: Mapping[str, tuple[float, ...]]


AUGMENTATION_REGISTRY: dict[str, AugmentationSpec] = {
    "identity": AugmentationSpec("identity", {}),
    "color_jitter": AugmentationSpec(
        "color_jitter",
        {
            "brightness": (0.05, 0.10, 0.20, 0.30, 0.40),
            "contrast": (0.05, 0.10, 0.20, 0.30, 0.40),
            "saturation": (0.00, 0.10, 0.20, 0.30, 0.40),
            "hue": (0.00, 0.02, 0.05, 0.08),
        },
    ),
    "gaussian_blur": AugmentationSpec("gaussian_blur", {"sigma": (0.15, 0.25, 0.50, 0.75, 1.00, 1.50)}),
    "grayscale": AugmentationSpec("grayscale", {"p": (1.0,)}),
    "noise": AugmentationSpec("noise", {"std": (0.01, 0.02, 0.03, 0.05, 0.08, 0.10)}),
    "random_resized_crop": AugmentationSpec(
        "random_resized_crop",
        {
            "scale_min": (0.80, 0.85, 0.90, 0.95),
            "scale_max": (1.0,),
            "ratio_min": (0.90, 0.95),
            "ratio_max": (1.05, 1.10),
        },
    ),
    "autocontrast": AugmentationSpec("autocontrast", {}),
    "sharpness": AugmentationSpec("sharpness", {"factor": (0.50, 0.75, 1.00, 1.25, 1.50)}),
    "posterize": AugmentationSpec("posterize", {"bits": (4, 5, 6, 7, 8)}),
    "solarize": AugmentationSpec("solarize", {"threshold": (64, 96, 128, 160, 192)}),
}

def summarize_augmentation_registry() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for operator, spec in AUGMENTATION_REGISTRY.items():
        summary[operator] = {
            "required_params": list(spec.grid.keys()),
            "legal_values": {key: [float(value) for value in values] for key, values in spec.grid.items()},
        }
    return summary


def _fmt(value: Any) -> str:
    return f"{float(value):.2f}"


def _fmt_int(value: Any) -> str:
    return str(int(round(float(value))))


def validate_view_spec(view: Mapping[str, Any]) -> dict[str, Any]:
    name = str(view.get("name") or "").strip()
    operator = str(view.get("operator") or "").strip()
    params = dict(view.get("params") or {})
    if not name:
        raise AugmentationError("view.name is required")
    if operator not in AUGMENTATION_REGISTRY:
        raise AugmentationError(f"unsupported augmentation operator: {operator}")
    spec = AUGMENTATION_REGISTRY[operator]
    expected = set(spec.grid)
    actual = set(params)
    if actual != expected:
        raise AugmentationError(f"{operator} params must be {sorted(expected)}, got {sorted(actual)}")
    for key, legal_values in spec.grid.items():
        value = float(params[key])
        if not any(abs(value - legal) < 1e-9 for legal in legal_values):
            raise AugmentationError(f"{operator}.{key}={value} is outside legal grid {list(legal_values)}")
        params[key] = value
    return {"name": name, "operator": operator, "params": params, "signature": view_signature(operator, params)}


def view_signature(operator: str, params: Mapping[str, Any] | None = None) -> str:
    params = dict(params or {})
    if operator == "identity":
        return "identity"
    if operator == "color_jitter":
        return f"color_jitter_b{_fmt(params['brightness'])}_c{_fmt(params['contrast'])}_s{_fmt(params['saturation'])}_h{_fmt(params['hue'])}"
    if operator == "gaussian_blur":
        return f"blur_sigma{_fmt(params['sigma'])}"
    if operator == "grayscale":
        return "grayscale_p1.00"
    if operator == "noise":
        return f"noise_std{_fmt(params['std'])}"
    if operator == "random_resized_crop":
        return f"crop_s{_fmt(params['scale_min'])}_{_fmt(params['scale_max'])}_r{_fmt(params['ratio_min'])}_{_fmt(params['ratio_max'])}"
    if operator == "autocontrast":
        return "autocontrast"
    if operator == "sharpness":
        return f"sharpness_f{_fmt(params['factor'])}"
    if operator == "posterize":
        return f"posterize_b{_fmt_int(params['bits'])}"
    if operator == "solarize":
        return f"solarize_t{_fmt_int(params['threshold'])}"
    raise AugmentationError(f"unsupported augmentation operator: {operator}")
