"""Concrete ClassificationHead implementations.

Each head maps a fused representation (B, in_dim) to logits (B, num_classes).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.modular.base import ClassificationHead


class MLPHead(ClassificationHead):
    """Two-layer MLP: Linear -> ReLU -> Dropout -> Linear.

    Migrated from ``layers.heads.MLPClassificationHead``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class LinearHead(ClassificationHead):
    """Single linear layer — minimal parameter count."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z: Tensor) -> Tensor:
        return self.fc(z)
