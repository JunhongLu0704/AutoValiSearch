from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticDomainSpec:
    domain_id: int
    split: str
    samples: int = 160
    image_size: int = 32
    num_classes: int = 3
    seed: int = 0


class SyntheticDomainDataset(Dataset):
    """Small domain-shifted image dataset for runnable public demos.

    Each class has a simple geometric signal. Each domain adds a different
    color/noise bias, so validation/test behavior is not identical to train.
    This keeps the demo fast while preserving the shape of a vision experiment.
    """

    def __init__(self, spec: SyntheticDomainSpec):
        self.spec = spec
        generator = torch.Generator().manual_seed(spec.seed + spec.domain_id * 1009)
        images = []
        labels = []
        domains = []
        for idx in range(spec.samples):
            label = idx % spec.num_classes
            images.append(_make_image(label, spec, generator))
            labels.append(label)
            domains.append(spec.domain_id)
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.domains = torch.tensor(domains, dtype=torch.long)

    def __len__(self) -> int:
        return self.spec.samples

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index], self.domains[index]


def _make_image(label: int, spec: SyntheticDomainSpec, generator: torch.Generator) -> torch.Tensor:
    size = spec.image_size
    image = torch.zeros(3, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")

    if label == 0:
        mask = (xx - size // 2).abs() < size // 8
    elif label == 1:
        mask = (yy - size // 2).abs() < size // 8
    else:
        mask = ((xx - yy).abs() < size // 10) | ((xx + yy - size).abs() < size // 10)

    class_color = torch.zeros(3)
    class_color[label % 3] = 0.8
    domain_phase = 2.0 * math.pi * spec.domain_id / 5.0
    domain_bias = torch.tensor(
        [
            0.12 * math.sin(domain_phase),
            0.12 * math.cos(domain_phase),
            0.08 * math.sin(domain_phase + 0.8),
        ]
    )
    image += 0.18 + domain_bias.view(3, 1, 1)
    image[:, mask] += class_color.view(3, 1)

    noise_scale = 0.06 if spec.split != "test" else 0.08
    image += torch.randn((3, size, size), generator=generator) * noise_scale
    return image.clamp(0.0, 1.0)


def build_synthetic_loaders(batch_size: int, seed: int, samples_per_split: int = 160):
    from torch.utils.data import DataLoader

    train = SyntheticDomainDataset(SyntheticDomainSpec(0, "train", samples_per_split, seed=seed))
    val = SyntheticDomainDataset(SyntheticDomainSpec(1, "val", samples_per_split // 2, seed=seed + 17))
    test = SyntheticDomainDataset(SyntheticDomainSpec(2, "test", samples_per_split // 2, seed=seed + 29))
    kwargs = {"batch_size": batch_size, "num_workers": 0}
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )
