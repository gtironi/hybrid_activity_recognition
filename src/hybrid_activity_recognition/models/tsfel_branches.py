"""Concrete TsfelBranch implementations.

Each branch consumes pre-computed TSFEL features (B, K) and returns an
embedding with shape (B, output_dim).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.base import TsfelBranch


class MLPTsfelBranch(TsfelBranch):
    """Linear projection of TSFEL features into the encoder embedding space.

    z_ts = LayerNorm(W x + b),  x in R^K, z_ts in R^{D_enc}.
  """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self._output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_features: Tensor) -> Tensor:
        return self.net(x_features)
