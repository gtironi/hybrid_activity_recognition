from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Pesos alinhados a índices 0..num_classes-1 (classes ausentes = peso 1.0)."""
    labels = labels.astype(np.int64)
    present = np.unique(labels)
    raw = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    full = np.ones(num_classes, dtype=np.float32)
    for i, c in enumerate(present):
        full[c] = raw[i]
    return torch.as_tensor(full, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weights.

    L = - weight_c * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)  # ce já incorpora o weight; p_t = softmax prob da classe correta
        focal = (1.0 - p_t).pow(self.gamma) * ce
        return focal.mean()


def supervised_loss_fn(
    class_weights: torch.Tensor | None,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
) -> nn.Module:
    if loss_type == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma)
    if loss_type != "ce":
        raise ValueError(f"Unknown loss_type={loss_type!r} (expected 'ce' or 'focal').")
    if class_weights is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=class_weights)
