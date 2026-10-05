"""Local pattern encoders, ordered memory, explicit task sharing and signal heads."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .redesign_model import CausalBlock, FusionResidual


@dataclass(frozen=True)
class MechanismConfig:
    prep: str = "dynamic"
    features: str = "selected"
    label: str = "joint"
    memory: str = "gru"
    sharing: str = "partial"
    width: int = 96
    slow_hours: int = 168
    minute: str = "none"
    modulation: bool = True
    horizon: int = 4
    path_frequency: int = 5
    fast_dim: int = 14
    seed: int = 20261004


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    valid = mask.any(-1, keepdim=True).to(values.dtype)
    return (values * valid).sum(1) / valid.sum(1).clamp_min(1)


class LocalEncoder(nn.Module):
    def __init__(self, dim: int, width: int, kernel: int = 5):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(dim * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.Sequential(CausalBlock(width, kernel, 1, .05), CausalBlock(width, kernel, 2, .05))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Mask nonfinite fields before projection and expose their identity.
        token = self.input(torch.cat((x * mask, mask), -1))
        return self.blocks(token) * mask.any(-1, keepdim=True)


class MemoryReadout(nn.Module):
    def __init__(self, width: int, kind: str):
        super().__init__()
        self.kind = kind
        self.recurrent = (nn.GRU(width, 64, batch_first=True) if kind == "gru" else
                          nn.LSTM(width, 64, batch_first=True) if kind == "lstm" else None)
        dim = width * (4 if kind == "ordered" else 3) + (64 if self.recurrent else 0)
        self.project = nn.Sequential(nn.Linear(dim, 96), nn.GELU(), nn.LayerNorm(96))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Recent 15m, 60m, and whole-window summaries; ordered quartiles distinguish older ages.
        if self.kind == "ordered":
            summaries = [masked_mean(v, m) for v, m in zip(x.chunk(4, 1), mask.chunk(4, 1))]
        else:
            summaries = [x[:, -1], masked_mean(x[:, -12:], mask[:, -12:]), masked_mean(x, mask)]
        if self.recurrent:
            sequence, _ = self.recurrent(x)  # reset to zero within each independent window
            summaries.append(sequence[:, -1])
        return self.project(torch.cat(summaries, -1))


class MechanismNetwork(nn.Module):
    def __init__(self, cfg: MechanismConfig):
        super().__init__()
        self.cfg = cfg
        self.fast = LocalEncoder(cfg.fast_dim + (6 if cfg.minute == "aggregate" else 0), cfg.width)
        self.memory = MemoryReadout(cfg.width, cfg.memory)
        self.slow = LocalEncoder(10, 48)
        self.slow_readout = nn.Sequential(nn.Linear(48 * 4, 64), nn.GELU(), nn.LayerNorm(64))
        self.context = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.LayerNorm(32))
        self.risk_context = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.LayerNorm(32)) if cfg.sharing == "isolated" else None
        self.minute = LocalEncoder(12, 48, 3) if cfg.minute == "encoder" else None
        self.minute_readout = nn.Sequential(nn.Linear(48 * 4, 48), nn.GELU(), nn.LayerNorm(48)) if self.minute else None
        fused = 96 + 64 + 32 + (48 if self.minute else 0)
        self.fusion = nn.Sequential(nn.Linear(fused, 128), nn.GELU(), nn.Dropout(.1),
                                    nn.LayerNorm(128), FusionResidual(128, .1))
        self.film = nn.Linear(32, 2 * cfg.width) if cfg.modulation else None
        if self.film:
            nn.init.normal_(self.film.weight, std=.001)
            nn.init.zeros_(self.film.bias)
        # Partial: common local features, separate risk aggregation, no signal-memory gradient from risk.
        # Isolated: independent local risk parameters. Full: auxiliaries use the final signal representation.
        self.risk_fast = LocalEncoder(cfg.fast_dim + (6 if cfg.minute == "aggregate" else 0), cfg.width) if cfg.sharing == "isolated" else None
        self.risk_readout = nn.Sequential(nn.Linear(cfg.width * 2 + 32, 128), nn.GELU(), nn.LayerNorm(128))
        self.signal = nn.Linear(128, 1)
        self.path = nn.Linear(128, 3)  # log(T), Q, A; the distinct objects remain explicit
        self.distribution = nn.Linear(128, 8)
        self.band = nn.Linear(128, 4)
        self.legacy_magnitude = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 4))
        self.sign = nn.Linear(128, 4)
        for head in (self.signal, self.path, self.distribution, self.band, self.sign):
            nn.init.normal_(head.weight, std=.002)
            nn.init.zeros_(head.bias)

    def shared_parameters(self):
        return list(self.fast.parameters())

    def forward(self, fast: Tensor, fm: Tensor, slow: Tensor, sm: Tensor,
                state: Tensor, cm: Tensor, minute: Tensor | None = None, mm: Tensor | None = None,
                representatives: Tensor | None = None) -> dict[str, Tensor]:
        context = self.context(torch.cat((state * cm, cm), -1))
        local = self.fast(fast, fm)
        signal_local = local
        if self.film:
            gamma, beta = (.1 * torch.tanh(self.film(context))).chunk(2, -1)
            signal_local = local * (1 + gamma[:, None]) + beta[:, None]
        memory = self.memory(signal_local, fm)
        sl = self.slow(slow, sm)
        back = self.slow_readout(torch.cat([masked_mean(v, m) for v, m in zip(sl.chunk(4, 1), sm.chunk(4, 1))], -1))
        pieces = [memory, back, context]
        if self.minute:
            ml = self.minute(minute, mm)
            # Only completed, ordered five-minute patches. No strided point sampling.
            patches = ml.reshape(len(ml), 36, 5, 48).mean(2)
            pieces.append(self.minute_readout(torch.cat([v.mean(1) for v in patches.chunk(4, 1)], -1)))
        h = self.fusion(torch.cat(pieces, -1))
        risk_local = self.risk_fast(fast, fm) if self.risk_fast else local
        risk_context = self.risk_context(torch.cat((state * cm, cm), -1)) if self.risk_context else context
        risk = h if self.cfg.sharing == "full" or self.cfg.label == "path" else self.risk_readout(torch.cat((
            masked_mean(risk_local, fm), risk_local[:, -1], risk_context), -1))
        path = self.path(risk)
        dist, band, sign = self.distribution(h), self.band(h), self.sign(h)
        if self.cfg.label == "four_bin":
            band = self.legacy_magnitude(torch.cat((state * cm, cm), -1))
        signal = self.signal(h).squeeze(-1)
        probability = torch.sigmoid(signal)
        if self.cfg.label == "path":
            signal = path[:, 1]  # Q proxy, intentionally not claimed to be an expected return
        elif self.cfg.label == "distribution":
            p = dist.softmax(-1)
            signal = (p * representatives).sum(-1)
            probability = p[:, 4:].sum(-1)
        elif self.cfg.label == "four_bin":
            q, pi = band.softmax(-1), sign.sigmoid()
            signal = (q * (pi * representatives[4:] + (1 - pi) * representatives[:4])).sum(-1)
            probability = (q * pi).sum(-1)
        return {"signal": signal, "path": path, "distribution": dist, "band": band,
                "sign": sign, "probability": probability, "representation": h,
                "local_activation": local}


class CausalLevelFilter(nn.Module):
    """A bounded, trainable convex combination with unit gain for constant inputs."""
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def alpha(self) -> Tensor:
        return .1 + .85 * torch.sigmoid(self.logit)

    def forward(self, values: Tensor) -> Tensor:
        alpha = self.alpha()
        state = values[0]
        result = [state]
        for row in values[1:]:
            state = (1 - alpha) * state + alpha * row
            result.append(state)
        return torch.stack(result)


def losses(output: dict, labels: Tensor, cfg: MechanismConfig) -> tuple[Tensor, Tensor]:
    # columns Y1/Y4/Y8, logT/Q/A. All target scales are fold-local and causal.
    horizon_index = {1: 0, 4: 1, 8: 2}[cfg.horizon]
    target = labels[:, horizon_index]
    path = F.mse_loss(output["path"], labels[:, 3:6])
    if cfg.label == "path":
        return path, path * 0
    if cfg.label == "distribution":
        return F.cross_entropy(output["distribution"], labels[:, 6].long()), path * 0
    if cfg.label == "four_bin":
        band = labels[:, 7].long()
        sign = torch.where(target == 0, torch.full_like(target, .5), (target > 0).float())
        primary = F.binary_cross_entropy_with_logits(output["sign"].gather(1, band[:, None]).squeeze(1), sign)
        return primary + F.cross_entropy(output["band"], band), path * 0
    primary = F.mse_loss(output["signal"], target)
    return primary, path if cfg.label == "joint" else path * 0
