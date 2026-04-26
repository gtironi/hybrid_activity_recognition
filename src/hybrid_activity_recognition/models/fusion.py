"""Concrete FusionModule implementations.

Each module merges signal and TSFEL embeddings into a single vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.base import FusionModule


class ConcatFusion(FusionModule):
    """Concatenation along the feature axis.

    Migrated from ``layers.fusion.ConcatFusion``.
    """

    def __init__(self, signal_dim: int, tsfel_dim: int):
        super().__init__()
        self._output_dim = signal_dim + tsfel_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, z_signal: Tensor, z_tsfel: Tensor) -> Tensor:
        return torch.cat((z_signal, z_tsfel), dim=1)
