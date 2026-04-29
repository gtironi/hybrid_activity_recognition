"""Augmentações FixMatch para sinal e vetor TSFEL (tensor)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SensorFixMatchAugment:
    def __init__(self, device: torch.device):
        self.device = device

    def _apply_3d_rotation(self, x: torch.Tensor, max_angle: float = 0.5) -> torch.Tensor:
        """Simulates collar shifting around the neck by rotating the 3D axis."""
        b, c, t = x.shape
        if c != 3:
            return x
            
        x_rot = x.clone()
        for i in range(b):
            # Random angles for pitch, roll, yaw
            theta = (torch.rand(3, device=self.device) - 0.5) * 2 * max_angle 
            
            rx = torch.tensor([[1, 0, 0],
                               [0, torch.cos(theta[0]), -torch.sin(theta[0])],
                               [0, torch.sin(theta[0]), torch.cos(theta[0])]], device=self.device)
            ry = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])],
                               [0, 1, 0],
                               [-torch.sin(theta[1]), 0, torch.cos(theta[1])]], device=self.device)
            rz = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0],
                               [torch.sin(theta[2]), torch.cos(theta[2]), 0],
                               [0, 0, 1]], device=self.device)
            
            R = rz @ ry @ rx
            x_rot[i] = torch.matmul(R, x[i])
        return x_rot

    def _apply_amplitude_scaling(self, x: torch.Tensor, min_scale: float = 0.7, max_scale: float = 1.3) -> torch.Tensor:
        """Simulates varying tightness of the collar strap."""
        b, c, t = x.shape
        scale = torch.rand(b, 1, 1, device=self.device) * (max_scale - min_scale) + min_scale
        return x * scale

    def weak_aug(self, x_signal: torch.Tensor, x_features: torch.Tensor):
        # Physically informed weak augmentations instead of generic noise
        aug_signal = self._apply_amplitude_scaling(x_signal, min_scale=0.85, max_scale=1.15)
        aug_signal = self._apply_3d_rotation(aug_signal, max_angle=0.2) # ~11 degrees rotation
        
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

        # Apply severe physical distortions
        x_aug = self._apply_amplitude_scaling(x_aug, min_scale=0.5, max_scale=1.5)
        x_aug = self._apply_3d_rotation(x_aug, max_angle=0.8) # ~45 degrees rotation

        f_aug = x_f + (torch.randn_like(x_f) * 0.05)
        return x_aug, f_aug

    def sa_strong_aug(self, x_signal: torch.Tensor, x_features: torch.Tensor):
        x_s, x_f = self.weak_aug(x_signal, x_features)
        b, c, t = x_s.shape
        x_aug = x_s.clone()

        # Semantic-Aware CutOut (SA-CutOut)
        for i in range(b):
            # 1. Channel-wise semantic masking (mask the axis with the highest variance)
            channel_vars = x_s[i].var(dim=1)
            dominant_channel = channel_vars.argmax().item()
            
            if np.random.rand() > 0.5:
                x_aug[i, dominant_channel, :] = 0.0
                
            # 2. Temporal semantic masking (mask the segment with the highest signal energy)
            mask_len = int(t * 0.3)
            if mask_len > 0:
                # Calculate absolute energy across channels
                energy = x_s[i].abs().sum(dim=0).unsqueeze(0).unsqueeze(0) 
                
                # Simple moving average to find the center of the highest energy cluster
                kernel = torch.ones(1, 1, mask_len, device=self.device)
                smoothed_energy = F.conv1d(energy, kernel, padding=mask_len//2).squeeze(0).squeeze(0)
                
                dominant_center = smoothed_energy.argmax().item()
                start = max(0, dominant_center - mask_len // 2)
                end = min(t, start + mask_len)
                
                x_aug[i, :, start:end] = 0.0

        # Add heavy physical augmentations alongside SA-CutOut
        x_aug = self._apply_amplitude_scaling(x_aug, min_scale=0.5, max_scale=1.5)
        x_aug = self._apply_3d_rotation(x_aug, max_angle=1.0) # ~57 degrees severe rotation

        f_aug = x_f + (torch.randn_like(x_f) * 0.05)
        return x_aug, f_aug