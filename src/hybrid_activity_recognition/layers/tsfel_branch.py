import torch.nn as nn


class TsfelMLPBranch(nn.Module):
    """Ramo TSFEL: projeção MLP + normalização + dropout."""

    def __init__(self, in_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
