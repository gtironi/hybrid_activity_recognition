"""Concrete FusionModule implementations.

Each module merges signal and TSFEL embeddings into a single vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.modular.base import FusionModule


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


class GatedCrossAttentionFusion(FusionModule):
    """Gated Cross-Attention (GCA) fusion mechanism.

    Uses TSFEL features to query deep signal embeddings via MultiheadAttention,
    then dynamically gates the contribution of the attended features versus the original.
    """

    def __init__(self, signal_dim: int, tsfel_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = signal_dim
        
        # Project TSFEL to match signal_dim if they differ
        if signal_dim != tsfel_dim:
            self.tsfel_proj = nn.Linear(tsfel_dim, signal_dim)
        else:
            self.tsfel_proj = nn.Identity()

        # Built-in PyTorch MultiheadAttention is highly optimized
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)

        # Learnable gate to evaluate feature reliability
        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Sigmoid()
        )

        # Output dimension remains the embedding dimension, as we fuse them mathematically
        self._output_dim = self.embed_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, z_signal: Tensor, z_tsfel: Tensor) -> Tensor:
        # Project TSFEL if needed
        z_t = self.tsfel_proj(z_tsfel)

        # Reshape to (Batch, SeqLen, EmbedDim) for Attention. We treat the pooled vector as SeqLen=1.
        q = z_t.unsqueeze(1)
        k = z_signal.unsqueeze(1)
        v = z_signal.unsqueeze(1)

        # Cross-Attention: TSFEL queries the Signal embedding
        attn_out, _ = self.mha(q, k, v)
        attn_out = attn_out.squeeze(1) # Back to (B, E)

        # Calculate dynamic gate weights based on both features
        gate_weights = self.gate(torch.cat((z_t, attn_out), dim=1))

        # Adaptive Feature Fusion Gating (AFFG)
        fused = (gate_weights * attn_out) + ((1 - gate_weights) * z_t)

        return fused