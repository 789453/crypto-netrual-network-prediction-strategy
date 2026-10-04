"""Independent direction and magnitude networks for the v3 return distribution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DirectionConfig:
    fast_dim: int = 16
    slow_dim: int = 10
    state_dim: int = 24
    fast_width: int = 96
    slow_width: int = 48
    fused_width: int = 96
    dropout_temporal: float = .05
    dropout_fusion: float = .10


class CausalBlock(nn.Module):
    def __init__(self, width: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.depthwise = nn.Conv1d(width, width, kernel, dilation=dilation,
                                   groups=width, bias=False)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        x = values.transpose(1, 2)
        update = self.pointwise(self.depthwise(F.pad(x, ((self.kernel - 1) * self.dilation, 0))))
        update = self.dropout(F.gelu(update)).transpose(1, 2)
        return self.norm(values + update)


def _masked_mean(values: Tensor, channel_mask: Tensor, recent: int | None = None) -> Tensor:
    if recent is not None:
        values = values[:, -recent:]
        channel_mask = channel_mask[:, -recent:]
    valid = (channel_mask.mean(dim=-1, keepdim=True) > .25).to(values.dtype)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class FusionResidual(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.net(x))


class DirectionNetwork(nn.Module):
    """Four conditional sign logits; no magnitude gradient enters this module."""

    def __init__(self, cfg: DirectionConfig | None = None, prior_logit: float = 0.0):
        super().__init__()
        self.cfg = cfg or DirectionConfig()
        c = self.cfg
        self.fast_input = nn.Sequential(nn.Linear(c.fast_dim * 2, c.fast_width), nn.GELU(),
                                        nn.LayerNorm(c.fast_width))
        self.fast_blocks = nn.Sequential(
            CausalBlock(c.fast_width, 9, 1, c.dropout_temporal),
            CausalBlock(c.fast_width, 9, 4, c.dropout_temporal),
        )
        self.fast_readout = nn.Sequential(nn.Linear(c.fast_width * 3, c.fast_width),
                                          nn.GELU(), nn.LayerNorm(c.fast_width))
        self.slow_input = nn.Sequential(nn.Linear(c.slow_dim * 2, c.slow_width), nn.GELU(),
                                        nn.LayerNorm(c.slow_width))
        self.slow_blocks = nn.Sequential(
            CausalBlock(c.slow_width, 7, 1, c.dropout_temporal),
            CausalBlock(c.slow_width, 7, 8, c.dropout_temporal),
        )
        self.slow_readout = nn.Sequential(nn.Linear(c.slow_width * 2, c.slow_width),
                                          nn.GELU(), nn.LayerNorm(c.slow_width))
        self.state_input = nn.Sequential(nn.Linear(c.state_dim * 2, 32), nn.GELU(), nn.LayerNorm(32))
        self.fusion = nn.Sequential(nn.Linear(c.fast_width + c.slow_width + 32, c.fused_width),
                                    nn.GELU(), nn.Dropout(c.dropout_fusion),
                                    nn.LayerNorm(c.fused_width),
                                    FusionResidual(c.fused_width, c.dropout_fusion))
        self.base = nn.Linear(c.fused_width, 1)
        self.offset = nn.Linear(c.fused_width, 4)
        nn.init.normal_(self.base.weight, std=.002)
        nn.init.constant_(self.base.bias, prior_logit)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, fast: Tensor, fast_mask: Tensor, slow: Tensor,
                slow_mask: Tensor, state: Tensor, state_mask: Tensor) -> Tensor:
        fx = self.fast_blocks(self.fast_input(torch.cat((fast, fast_mask), dim=-1)))
        f = self.fast_readout(torch.cat((fx[:, -1], _masked_mean(fx, fast_mask, 12),
                                         _masked_mean(fx, fast_mask)), dim=-1))
        sx = self.slow_blocks(self.slow_input(torch.cat((slow, slow_mask), dim=-1)))
        s = self.slow_readout(torch.cat((sx[:, -1], _masked_mean(sx, slow_mask)), dim=-1))
        context = self.state_input(torch.cat((state, state_mask), dim=-1))
        shared = self.fusion(torch.cat((f, s, context), dim=-1))
        return self.base(shared) + self.offset(shared)


class MagnitudeNetwork(nn.Module):
    """Small state-only classifier with its own optimizer and checkpoint."""

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim * 2, 32), nn.GELU(), nn.LayerNorm(32),
                                 nn.Linear(32, 4))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.net(torch.cat((values, mask), dim=-1))


def direction_loss(logits: Tensor, band: Tensor, up: Tensor, offset_weight: Tensor | None = None) -> Tensor:
    selected = logits.gather(1, band.long().unsqueeze(1)).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(selected.float(), up.float())
    if offset_weight is not None:
        loss = loss + .001 * torch.square(offset_weight).mean()
    return loss


def magnitude_loss(logits: Tensor, band: Tensor) -> Tensor:
    return F.cross_entropy(logits.float(), band.long())
