from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedConv2dWrapper(nn.Module):
    def __init__(self, conv: nn.Conv2d, bit_width: int):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError('QuantizedConv2dWrapper expects an nn.Conv2d module')
        if bit_width not in (4, 8):
            raise ValueError(f'Unsupported bit width: {bit_width}')

        self.conv = conv
        self.bit_width = int(bit_width)
        self.register_buffer('_enabled', torch.tensor(False, dtype=torch.bool))

        qmax = (1 << (self.bit_width - 1)) - 1
        init_scale = conv.weight.detach().abs().max()
        if not torch.isfinite(init_scale) or init_scale <= 0:
            init_scale = torch.tensor(1.0, dtype=conv.weight.dtype, device=conv.weight.device)
        else:
            init_scale = init_scale / max(qmax, 1)
        self.step_size = nn.Parameter(init_scale.reshape(1))

    @property
    def enabled(self) -> bool:
        return bool(self._enabled.item())

    def set_enabled(self, enabled: bool = True) -> None:
        self._enabled.fill_(bool(enabled))

    def _fake_quantize_weight(self) -> torch.Tensor:
        if not self.enabled:
            return self.conv.weight

        qmax = (1 << (self.bit_width - 1)) - 1
        qmin = -(1 << (self.bit_width - 1))
        scale = self.step_size.abs().clamp_min(1e-8)
        scaled = self.conv.weight / scale
        clipped = scaled.clamp(qmin, qmax)
        rounded = clipped + (torch.round(clipped) - clipped).detach()
        return rounded * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._fake_quantize_weight()
        return F.conv2d(
            x,
            weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


class QATManager:
    def __init__(self, wrappers: List[QuantizedConv2dWrapper], scope: str, bit_width: int):
        self.wrappers = wrappers
        self.scope = scope
        self.bit_width = int(bit_width)
        self._enabled = False

    def maybe_enable(self, global_step: int, total_steps: int, start_ratio: float) -> bool:
        if self._enabled:
            return False
        if total_steps <= 0:
            return False

        progress = float(global_step) / float(total_steps)
        if progress < start_ratio:
            return False

        for wrapper in self.wrappers:
            wrapper.set_enabled(True)
        self._enabled = True
        return True

    def is_enabled(self) -> bool:
        return self._enabled

    def num_wrapped_convs(self) -> int:
        return len(self.wrappers)


_SCOPE_TO_STAGES = {
    'layer4': ('layer4',),
    'layer3_4': ('layer3', 'layer4'),
    'layer2_4': ('layer2', 'layer3', 'layer4'),
}


def resolve_quant_scope(model: nn.Module, scope: str) -> List[nn.Module]:
    if scope not in _SCOPE_TO_STAGES:
        raise ValueError(f'Unsupported quant_scope: {scope}')

    stages = []
    for stage_name in _SCOPE_TO_STAGES[scope]:
        if not hasattr(model, stage_name):
            raise ValueError(f'Model has no stage named {stage_name}')
        stages.append(getattr(model, stage_name))
    return stages


def _wrap_stage_convs(module: nn.Module, bit_width: int, wrappers: List[QuantizedConv2dWrapper]) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, QuantizedConv2dWrapper):
            wrappers.append(child)
            continue
        if isinstance(child, nn.Conv2d):
            wrapped = QuantizedConv2dWrapper(child, bit_width)
            setattr(module, child_name, wrapped)
            wrappers.append(wrapped)
            continue
        _wrap_stage_convs(child, bit_width, wrappers)


def attach_qat_wrappers(model: nn.Module, scope: str, bit_width: int) -> QATManager:
    wrappers: List[QuantizedConv2dWrapper] = []
    for stage in resolve_quant_scope(model, scope):
        _wrap_stage_convs(stage, bit_width, wrappers)
    if not wrappers:
        raise ValueError(f'No Conv2d modules were wrapped for quant_scope={scope}')
    return QATManager(wrappers=wrappers, scope=scope, bit_width=bit_width)
