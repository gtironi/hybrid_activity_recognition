"""HybridModel — container that wires the four modular components.

Supports two modes:
- ``deep_only``:  encoder → head  (TSFEL branch and fusion are None)
- ``hybrid``:     encoder + TSFEL branch → fusion → head

The forward signature ``(x_signal, x_features) → logits`` is identical in both
modes so the Trainer does not need to know which mode is active.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.modular.base import (
    ClassificationHead,
    FusionModule,
    SignalEncoder,
    TsfelBranch,
)


class HybridModel(nn.Module):
    """Modular hybrid model container."""

    def __init__(
        self,
        encoder: SignalEncoder,
        tsfel_branch: TsfelBranch | None,
        fusion: FusionModule | None,
        head: ClassificationHead,
        input_mode: Literal["deep_only", "hybrid"] = "hybrid",
    ):
        super().__init__()
        self.input_mode = input_mode
        self.encoder = encoder
        self.tsfel_branch = tsfel_branch
        self.fusion = fusion
        self.head = head

    def forward(self, x_signal: Tensor, x_features: Tensor) -> Tensor:
        z_sig = self.encoder(x_signal)

        if self.input_mode == "deep_only":
            return self.head(z_sig)

        z_ts = self.tsfel_branch(x_features)
        z_fused = self.fusion(z_sig, z_ts)
        return self.head(z_fused)
