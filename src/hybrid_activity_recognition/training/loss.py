from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
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


def supervised_loss_fn(class_weights: torch.Tensor | None) -> nn.Module:
    if class_weights is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=class_weights)
