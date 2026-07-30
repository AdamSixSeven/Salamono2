from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PoseTCN(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        channels: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_features, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(channels, dilation=1, dropout=dropout),
            ResidualTemporalBlock(channels, dilation=2, dropout=dropout),
            ResidualTemporalBlock(channels, dilation=4, dropout=dropout),
            ResidualTemporalBlock(channels, dilation=8, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: batch, time, features.
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.head(x)
