from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_dim: int = 387
    tcn_channels: int = 128
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.20
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    action_classes: int = 9
    safety_classes: int = 4

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dilations"] = list(self.dilations)
        return d

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        value = dict(value)
        value["dilations"] = tuple(value.get("dilations", (1, 2, 4, 8)))
        return cls(**value)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        return F.gelu(x + residual)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TCNGRUModel(nn.Module):
    """Past-only input window -> local TCN patterns -> GRU event context -> 2 heads."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.Linear(config.input_dim, config.tcn_channels),
            nn.LayerNorm(config.tcn_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.tcn = nn.ModuleList([
            ResidualTCNBlock(config.tcn_channels, config.kernel_size, d, config.dropout)
            for d in config.dilations
        ])
        self.gru = nn.GRU(
            input_size=config.tcn_channels,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attention = TemporalAttention(config.gru_hidden)
        fused = config.gru_hidden + config.tcn_channels
        self.shared = nn.Sequential(
            nn.Linear(fused, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.action_head = nn.Linear(192, config.action_classes)
        self.safety_head = nn.Linear(192, config.safety_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: B,T,F
        encoded = self.input(x)
        tcn = encoded.transpose(1, 2)
        for block in self.tcn:
            tcn = block(tcn)
        tcn_t = tcn.transpose(1, 2)
        gru_out, _ = self.gru(tcn_t)
        context = self.attention(gru_out, mask=mask)
        if mask is None:
            local = tcn_t[:, -1]
        else:
            lengths = mask.long().sum(dim=1).clamp_min(1) - 1
            local = tcn_t[torch.arange(len(tcn_t), device=tcn_t.device), lengths]
        z = self.shared(torch.cat([context, local], dim=1))
        return self.action_head(z), self.safety_head(z)
