import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Fusão por concatenação no eixo de features."""

    def forward(self, signal_embedding: torch.Tensor, tsfel_embedding: torch.Tensor) -> torch.Tensor:
        return torch.cat((signal_embedding, tsfel_embedding), dim=1)
