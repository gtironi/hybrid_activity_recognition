"""Augmentações FixMatch para sinal e vetor TSFEL (tensor)."""

from __future__ import annotations

import numpy as np
import torch


class SensorFixMatchAugment:
    def __init__(self, device: torch.device):
        self.device = device

    def weak_aug(self, x_signal: torch.Tensor, x_features: torch.Tensor):
        b, _, _ = x_signal.shape
        noise = torch.randn_like(x_signal) * 0.05
        scale = torch.rand(b, 1, 1, device=self.device) * 0.2 + 0.9
        aug_signal = (x_signal * scale) + noise
        aug_features = x_features + (torch.randn_like(x_features) * 0.02)
        return aug_signal, aug_features

    def strong_aug(self, x_signal: torch.Tensor, x_features: torch.Tensor):
        x_s, x_f = self.weak_aug(x_signal, x_features)
        b, c, t = x_s.shape
        x_aug = x_s.clone()

        if np.random.rand() > 0.5:
            if t > 10:
                num_segments = np.random.randint(2, 5)
                seg_len = t // num_segments
                for i in range(b):
                    perm = torch.randperm(num_segments)
                    parts = [x_s[i, :, p * seg_len : (p + 1) * seg_len] for p in perm]
                    shuffled = torch.cat(parts, dim=1)
                    if shuffled.shape[1] < t:
                        pad = torch.zeros(c, t - shuffled.shape[1], device=self.device)
                        shuffled = torch.cat([shuffled, pad], dim=1)
                    elif shuffled.shape[1] > t:
                        shuffled = shuffled[:, :t]
                    x_aug[i] = shuffled
        else:
            mask_len = int(t * 0.3)
            for i in range(b):
                start = int(np.random.randint(0, max(1, t - mask_len)))
                x_aug[i, :, start : start + mask_len] = 0.0

        f_aug = x_f + (torch.randn_like(x_f) * 0.05)
        return x_aug, f_aug
