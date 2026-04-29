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
    """
    Focal Loss to address extreme long-tail distributions in activity datasets.
    Down-weights well-classified examples to focus on hard, rare examples.
    """
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def supervised_loss_fn(class_weights: torch.Tensor | None, use_focal_loss: bool = False) -> nn.Module:
    if use_focal_loss:
        return FocalLoss(weight=class_weights, gamma=2.0)
    if class_weights is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=class_weights)


def log_magnitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Frequency-aware Log-scale Mean-Magnitude (LMM) loss."""
    # HF format is (B, T, C), we compute FFT along the time dimension (dim=1)
    fft_pred = torch.fft.rfft(pred, dim=1)
    fft_target = torch.fft.rfft(target, dim=1)
    
    # log(|F(x)| + 1e-8) to stabilize early training
    log_mag_pred = torch.log(torch.abs(fft_pred) + 1e-8)
    log_mag_target = torch.log(torch.abs(fft_target) + 1e-8)
    
    return F.mse_loss(log_mag_pred, log_mag_target)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE Contrastive loss between two augmented views."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    
    return F.cross_entropy(sim_matrix, labels)