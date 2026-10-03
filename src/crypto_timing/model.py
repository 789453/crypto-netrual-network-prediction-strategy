"""Small, deep causal timing network adapted from BigAlpha's TCN/stat paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    fast_dim: int = 31
    slow_dim: int = 20
    stats_dim: int = 80
    market_dim: int = 10
    hidden: int = 64
    fused: int = 128
    dropout: float = 0.1
    fast_blocks: int = 3
    use_slow: bool = True
    use_stats: bool = True
    use_market: bool = True


class CausalMultiScaleBlock(nn.Module):
    def __init__(self, dim: int, kernels: tuple[int, ...], dilation: int, dropout: float):
        super().__init__()
        self.kernels = kernels
        self.dilation = dilation
        self.branches = nn.ModuleList(
            nn.Conv1d(dim, dim, k, dilation=dilation, groups=dim, bias=False)
            for k in kernels
        )
        self.mix = nn.Conv1d(dim * len(kernels), dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        channels = x.transpose(1, 2)
        branches = [
            conv(F.pad(channels, ((k - 1) * self.dilation, 0)))
            for conv, k in zip(self.branches, self.kernels, strict=True)
        ]
        update = self.dropout(F.gelu(self.mix(torch.cat(branches, dim=1)))).transpose(1, 2)
        return self.norm(x + update)


class AttentionSummary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.logit = nn.Linear(dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return (x * torch.softmax(self.logit(x).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.net(x))


class CryptoTimingNetwork(nn.Module):
    """Outputs (half-variance logit, volatility, terminal risk-return)."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        d = cfg.hidden
        self.fast_project = nn.Sequential(nn.Linear(cfg.fast_dim * 2, d), nn.GELU(), nn.LayerNorm(d))
        self.fast_blocks = nn.ModuleList(
            CausalMultiScaleBlock(d, (3, 9, 25), 2**i, cfg.dropout)
            for i in range(cfg.fast_blocks)
        )
        self.fast_attention = AttentionSummary(d)
        self.fast_merge = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.LayerNorm(d))
        if cfg.use_slow:
            self.slow_project = nn.Sequential(nn.Linear(cfg.slow_dim * 2, d), nn.GELU(), nn.LayerNorm(d))
            self.slow_blocks = nn.ModuleList(
                CausalMultiScaleBlock(d, (3, 9), dilation, cfg.dropout)
                for dilation in (1, 4)
            )
            self.slow_attention = AttentionSummary(d)
            self.slow_merge = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))
        if cfg.use_stats:
            self.stats_path = nn.Sequential(
                nn.Linear(cfg.stats_dim * 2, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.LayerNorm(d)
            )
        if cfg.use_market:
            self.market_path = nn.Sequential(
                nn.Linear(cfg.market_dim * 2, 32), nn.GELU(), nn.LayerNorm(32)
            )
        branch_count = 1 + int(cfg.use_slow) + int(cfg.use_stats)
        self.gate = nn.Linear(d + (32 if cfg.use_market else 0), branch_count)
        self.fusion = nn.Sequential(
            nn.Linear(branch_count * d + (32 if cfg.use_market else 0), cfg.fused),
            nn.GELU(), nn.Dropout(cfg.dropout), nn.LayerNorm(cfg.fused),
            ResidualMLP(cfg.fused, cfg.dropout), ResidualMLP(cfg.fused, cfg.dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.fused, 64), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(64, 3)
        )

    def forward(self, fast: Tensor, fast_mask: Tensor, slow: Tensor, slow_mask: Tensor,
                stats: Tensor, stats_mask: Tensor, market: Tensor, market_mask: Tensor) -> Tensor:
        fast_x = self.fast_project(torch.cat((fast, fast_mask), dim=-1))
        for block in self.fast_blocks:
            fast_x = block(fast_x)
        fast_summary = self.fast_merge(torch.cat((
            fast_x[:, -1], fast_x.mean(dim=1), self.fast_attention(fast_x),
            fast_x[:, -48:].mean(dim=1),
        ), dim=-1))
        branches = [fast_summary]
        if self.config.use_slow:
            slow_x = self.slow_project(torch.cat((slow, slow_mask), dim=-1))
            for block in self.slow_blocks:
                slow_x = block(slow_x)
            branches.append(self.slow_merge(torch.cat((
                slow_x[:, -1], slow_x.mean(dim=1), self.slow_attention(slow_x),
            ), dim=-1)))
        if self.config.use_stats:
            branches.append(self.stats_path(torch.cat((stats, stats_mask), dim=-1)))
        context = self.market_path(torch.cat((market, market_mask), dim=-1)) if self.config.use_market else None
        gate_input = torch.cat((fast_summary, context), dim=-1) if context is not None else fast_summary
        weights = torch.softmax(self.gate(gate_input), dim=-1)
        gated = torch.cat([branch * weights[:, i : i + 1] for i, branch in enumerate(branches)], dim=-1)
        fused = torch.cat((gated, context), dim=-1) if context is not None else gated
        return self.head(self.fusion(fused))
