"""Concrete TsfelBranch implementations.

Each branch consumes pre-computed TSFEL features (B, K) and returns an
embedding with shape (B, output_dim).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.base import TsfelBranch


class MLPTsfelBranch(TsfelBranch):
    """Identity TSFEL branch.

    Returns the input tensor unchanged so TSFEL features flow directly into fusion or the head.
    """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self._output_dim = in_features

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_features: Tensor) -> Tensor:
        return x_features


class DeepMLPTsfelBranch(TsfelBranch):
    """Three-layer MLP with BatchNorm, ReLU and Dropout per stage.

    Reproduces the ``mlp`` block of ``TsfelOnlyModel`` from
    ``actbecalf-windowed.ipynb``: ``n_feats -> 256 -> 128 -> 64`` with
    BN + ReLU + Dropout after every Linear, and Kaiming initialization.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self._output_dim = 64
        self.mlp = nn.Sequential(
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
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_features: Tensor) -> Tensor:
        return self.mlp(x_features)
