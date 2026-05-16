from __future__ import annotations

import torch


def stable_sample_weights(features: torch.Tensor, lambdap: float, num_f: int) -> torch.Tensor:
    """Compute simple demo sample weights from feature stability.

    This is a lightweight public analogue of stable reweighting: high feature
    norm outliers receive lower weights, and the strength is controlled by
    `lambdap`. `num_f` controls how many feature chunks are used to estimate the
    per-sample instability score.
    """

    if features.ndim != 2:
        raise ValueError("features must be [batch, dim]")
    chunks = torch.chunk(features.detach(), max(1, int(num_f)), dim=1)
    penalties = []
    for chunk in chunks:
        centered = chunk - chunk.mean(dim=0, keepdim=True)
        penalties.append(centered.pow(2).mean(dim=1))
    penalty = torch.stack(penalties, dim=1).mean(dim=1)
    penalty = penalty / (penalty.mean().clamp_min(1e-6))
    weights = torch.exp(-float(lambdap) * 0.03 * penalty)
    return weights / weights.mean().clamp_min(1e-6)
