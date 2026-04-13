"""Concrete TsfelBranch implementations.

Each branch projects pre-computed TSFEL features (B, K) to (B, output_dim).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.modular.base import TsfelBranch


class MLPTsfelBranch(TsfelBranch):
    """Single hidden-layer MLP projection: Linear -> BN -> ReLU -> Dropout.

    Migrated from ``layers.tsfel_branch.TsfelMLPBranch``.
    """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self._output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_features: Tensor) -> Tensor:
        return self.net(x_features)
