"""Online signal augmentations for supervised training (train split only)."""

from __future__ import annotations

import torch
from torch import Tensor


class SignalAugmentation:
    """Stochastic perturbations on raw accelerometer windows ``(C, T)``."""

    def __init__(
        self,
        jitter_std: float = 0.03,
        scale_mean: float = 1.0,
        scale_std: float = 0.1,
    ):
        self.jitter_std = jitter_std
        self.scale_mean = scale_mean
        self.scale_std = scale_std

    def __call__(self, x: Tensor) -> Tensor:
        """x: (C, T) float tensor."""
        if self.jitter_std > 0:
            x = x + torch.randn_like(x) * self.jitter_std
        if self.scale_std > 0:
            scale = torch.randn(x.shape[0], 1, device=x.device, dtype=x.dtype)
            scale = scale * self.scale_std + self.scale_mean
            x = x * scale
        return x
