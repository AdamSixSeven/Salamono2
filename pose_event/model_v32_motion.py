from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


class ResidualTCNBlockV32(nn.Module):
    """Residual TCN block used by the V3.2 motion-only training code."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(x + self.net(x))


class TemporalBranchV32(nn.Module):
    """Exact temporal encoder used by the V3.2 train-normalization ablation."""

    def __init__(
        self,
        *,
        input_dim: int,
        tcn_channels: int,
        gru_hidden: int,
        gru_layers: int,
        shared_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, tcn_channels),
            nn.LayerNorm(tcn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(
            ResidualTCNBlockV32(tcn_channels, 1, dropout),
            ResidualTCNBlockV32(tcn_channels, 2, dropout),
            ResidualTCNBlockV32(tcn_channels, 4, dropout),
            ResidualTCNBlockV32(tcn_channels, 8, dropout),
        )
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.shared = nn.Sequential(
            nn.Linear(gru_hidden + tcn_channels, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        x = x * mask.unsqueeze(-1).to(x.dtype)

        x = self.input_projection(x)
        tcn_seq = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        gru_seq, _ = self.gru(tcn_seq)

        attn_scores = self.attention(gru_seq).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(gru_seq * attn_weights.unsqueeze(-1), dim=1)

        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
        last_index = torch.where(
            mask,
            positions,
            torch.full_like(positions, -1),
        ).max(dim=1).values.clamp_min(0)
        batch_index = torch.arange(x.shape[0], device=x.device)
        last_tcn = tcn_seq[batch_index, last_index]

        return self.shared(torch.cat([pooled, last_tcn], dim=1))


@dataclass
class V32MotionConfig:
    input_dim: int = 387
    tcn_channels: int = 128
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.20
    shared_dim: int = 192
    action_classes: int = 9
    safety_classes: int = 4

    @classmethod
    def from_dict(cls, value: dict) -> "V32MotionConfig":
        allowed = {
            "input_dim",
            "tcn_channels",
            "gru_hidden",
            "gru_layers",
            "dropout",
            "shared_dim",
            "action_classes",
            "safety_classes",
        }
        return cls(**{key: value[key] for key in allowed if key in value})


class V32MotionModel(nn.Module):
    """Action+safety inference wrapper for a V3.2 motion-only checkpoint."""

    def __init__(self, config: V32MotionConfig):
        super().__init__()
        self.config = config
        self.motion_branch = TemporalBranchV32(
            input_dim=config.input_dim,
            tcn_channels=config.tcn_channels,
            gru_hidden=config.gru_hidden,
            gru_layers=config.gru_layers,
            shared_dim=config.shared_dim,
            dropout=config.dropout,
        )
        self.action_head = nn.Linear(config.shared_dim, config.action_classes)
        self.safety_head = nn.Linear(config.shared_dim, config.safety_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        h = self.motion_branch(x, mask)
        return self.action_head(h), self.safety_head(h)
