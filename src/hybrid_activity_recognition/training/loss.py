from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Weights aligned to indices 0..num_classes-1 (absent classes → weight 1.0)."""
    labels = labels.astype(np.int64)
    present = np.unique(labels)
    raw = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    full = np.ones(num_classes, dtype=np.float32)
    for i, c in enumerate(present):
        full[c] = raw[i]
    return torch.as_tensor(full, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t).

    ``alpha`` may be a scalar or per-class tensor (C,) broadcast over the batch.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | float | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        elif alpha is None:
            self.alpha = None
        else:
            self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[targets]
            else:
                alpha_t = self.alpha
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_supervised_criterion(
    loss_criterion: str,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    *,
    use_class_weights: bool = True,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Stage-2 criterion: weighted CE or focal loss with optional class balancing."""
    if loss_criterion == "weighted_ce":
        cw = None
        if use_class_weights:
            cw = balanced_class_weights(labels, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=cw)

    if loss_criterion == "focal":
        alpha = None
        if use_class_weights:
            alpha = balanced_class_weights(labels, num_classes).to(device)
        return FocalLoss(gamma=focal_gamma, alpha=alpha).to(device)

    raise ValueError(
        f"Unknown loss_criterion: {loss_criterion!r}. Use 'weighted_ce' or 'focal'."
    )
