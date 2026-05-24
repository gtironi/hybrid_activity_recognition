"""Concrete FusionModule implementations.

Each module merges signal and TSFEL embeddings into a single vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.base import FusionModule


class ConcatFusion(FusionModule):
    """Concatenation along the feature axis (legacy baseline)."""

    def __init__(self, signal_dim: int, tsfel_dim: int, d_fused: int | None = None):
        super().__init__()
        del d_fused  # unused; output is always signal_dim + tsfel_dim
        self._signal_dim = signal_dim
        self._tsfel_dim = tsfel_dim
        self._output_dim = signal_dim + tsfel_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, z_signal: Tensor, z_tsfel: Tensor) -> Tensor:
        return torch.cat((z_signal, z_tsfel), dim=1)


class GatedFusion(FusionModule):
    """Element-wise GMU / GIF-style fusion in a shared latent space.

    Requires ``d_encoder == d_tsfel`` (both branches projected to D_enc).

    Gate (per sample, per dimension):
        g = sigmoid(W_g [z_sig ; z_ts] + b_g)   in (0, 1)^{B x D}

    Fusion:
        z_fused = g * z_sig + (1 - g) * z_tsfel
    """

    def __init__(self, d_encoder: int, d_tsfel: int, d_fused: int | None = None):
        super().__init__()
        if d_tsfel != d_encoder:
            raise ValueError(
                f"GatedFusion expects aligned embeddings: d_encoder={d_encoder}, d_tsfel={d_tsfel}"
            )
        d_out = d_fused if d_fused is not None else d_encoder
        if d_out != d_encoder:
            raise ValueError(
                f"GatedFusion output dim must match encoder dim ({d_encoder}), got d_fused={d_out}"
            )
        self._output_dim = d_out
        self.gate = nn.Linear(2 * d_encoder, d_encoder)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, z_signal: Tensor, z_tsfel: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(torch.cat((z_signal, z_tsfel), dim=1)))
        return g * z_signal + (1.0 - g) * z_tsfel
