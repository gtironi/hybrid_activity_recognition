"""HybridModel — container that wires the four modular components.

Supports three modes:
- ``deep_only``:  encoder → head  (TSFEL branch and fusion are None)
- ``hybrid``:     encoder + TSFEL branch → fusion → head
- ``tsfel_only``: TSFEL branch → head (signal encoder ignored)

Optional subject-adversarial branch: encoder embedding → GRL → SubjectDiscriminator.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.base import (
    ClassificationHead,
    FusionModule,
    SignalEncoder,
    TsfelBranch,
)
from hybrid_activity_recognition.models.grl import GradientReversal


class HybridModel(nn.Module):
    """Modular hybrid model container."""

    def __init__(
        self,
        encoder: SignalEncoder,
        tsfel_branch: TsfelBranch | None,
        fusion: FusionModule | None,
        head: ClassificationHead,
        input_mode: Literal["deep_only", "hybrid", "tsfel_only"] = "hybrid",
        subject_discriminator: nn.Module | None = None,
    ):
        super().__init__()
        self.input_mode = input_mode
        self.encoder = encoder
        self.tsfel_branch = tsfel_branch
        self.fusion = fusion
        self.head = head
        self.subject_discriminator = subject_discriminator
        self.grl = GradientReversal()

    def forward(
        self,
        x_signal: Tensor,
        x_features: Tensor,
        *,
        compute_adversarial: bool = False,
        grl_alpha: float = 1.0,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.input_mode == "tsfel_only":
            behaviour_logits = self.head(self.tsfel_branch(x_features))
            if compute_adversarial:
                raise ValueError("Subject adversarial alignment requires a signal encoder.")
            return behaviour_logits

        z_sig = self.encoder(x_signal)

        if self.input_mode == "deep_only":
            if getattr(self.head, "needs_patchtst_hidden", False):
                z_head_in = self.encoder.forward_hidden(x_signal)
            else:
                z_head_in = z_sig
            behaviour_logits = self.head(z_head_in)
        else:
            z_ts = self.tsfel_branch(x_features)
            z_fused = self.fusion(z_sig, z_ts)
            behaviour_logits = self.head(z_fused)

        if compute_adversarial:
            if self.subject_discriminator is None:
                raise ValueError("compute_adversarial=True but subject_discriminator is None.")
            z_rev = self.grl(z_sig, grl_alpha)
            subject_logits = self.subject_discriminator(z_rev)
            return behaviour_logits, subject_logits

        return behaviour_logits
