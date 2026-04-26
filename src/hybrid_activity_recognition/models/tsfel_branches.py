"""Concrete TsfelBranch implementations.

Each branch consumes pre-computed TSFEL features (B, K) and returns an
embedding with shape (B, output_dim).
"""

from __future__ import annotations

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
