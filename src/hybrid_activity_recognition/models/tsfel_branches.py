"""Concrete TsfelBranch implementations.

Each branch consumes pre-computed TSFEL features (B, K) and returns an
embedding with shape (B, output_dim).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.base import TsfelBranch


class MLPTsfelBranch(TsfelBranch):
    """3-layer MLP with BatchNorm: 256 → 128 → 64, Dropout 0.4 after each block.

    Matches the TsfelOnlyModel MLP from the notebook baseline.
    Output dim is always 64 regardless of in_features / hidden_dim args.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self._output_dim = 64
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_features: Tensor) -> Tensor:
        return self.net(x_features)
