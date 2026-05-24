"""Gradient Reversal Layer (Ganin et al.) for domain / subject adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Reverses gradients scaled by ``alpha`` during backpropagation."""

    def forward(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        return _GradientReversalFunction.apply(x, alpha)
