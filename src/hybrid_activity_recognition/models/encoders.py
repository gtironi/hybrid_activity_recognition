"""Concrete SignalEncoder implementations.

Each encoder maps (B, C, T) sensor windows to (B, output_dim) embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.base import SignalEncoder


class TemporalAttentionPooling(nn.Module):
    """Attention pooling over LSTM time steps. Input (B, T, D) -> (B, D)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, seq: Tensor) -> Tensor:
        b, _t, d = seq.shape
        scale = float(d) ** 0.5
        scores = torch.matmul(self.query.expand(b, -1, -1), seq.transpose(1, 2)) / scale
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, seq).squeeze(1)
        return self.proj(pooled)


class PatchTokenAttentionPooling(nn.Module):
    """Attention pooling over channel x patch tokens from PatchTST hidden states.

    Input:  (B, C, P, D)
    Output: (B, D)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: Tensor) -> Tensor:
        # (B, C, P, D) -> (B, N, D) with N = C * P
        b, _c, _p, d = hidden.shape
        tokens = hidden.reshape(b, -1, d)
        scale = float(d) ** 0.5
        scores = torch.matmul(self.query.expand(b, -1, -1), tokens.transpose(1, 2)) / scale
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, tokens).squeeze(1)
        return self.proj(pooled)


class CNNLSTMEncoder(SignalEncoder):
    """2 Conv1D blocks + 2-layer BiLSTM, last timestep aggregation.

    Migrated from ``layers.signal_branch.HybridCNNLSTMSignalBranch``.
    Default output_dim = 2 * hidden_lstm = 128.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_lstm: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self._output_dim = hidden_lstm * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_signal: Tensor) -> Tensor:
        x = self.cnn(x_signal)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]


class RobustCNNLSTMEncoder(SignalEncoder):
    """3 Conv1D blocks (single temporal downsample) + BiLSTM + temporal attention pool.

    For T=75, one ``MaxPool1d(2)`` yields ~37 steps into the LSTM (vs ~18 with two pools).
    Default output_dim = 2 * hidden_lstm = 256.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_lstm: int = 128,
        cnn_dropout: float = 0.4,
        lstm_dropout: float = 0.4,
    ):
        super().__init__()
        self._output_dim = hidden_lstm * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_lstm,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.temporal_pool = TemporalAttentionPooling(self._output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_signal: Tensor) -> Tensor:
        x = self.cnn(x_signal)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        return self.temporal_pool(lstm_out)


class PatchTSTEncoder(SignalEncoder):
    """HuggingFace ``PatchTSTModel`` + attention pooling over patch tokens.

    ``patch_length`` and ``patch_stride`` control overlapping token grids
    (e.g. T=75, patch=15, stride=5 -> ~13 tokens). Hidden states
    ``(B, C, P, D)`` are aggregated with ``PatchTokenAttentionPooling`` instead
    of a global mean.

    If ``pretrained_path`` is given, loads backbone weights from MAE pretraining
    (pooler weights are trained from scratch in supervised stages).
    """

    def __init__(
        self,
        context_length: int = 75,
        patch_length: int = 12,
        patch_stride: int = 12,
        d_model: int = 128,
        num_heads: int = 16,
        num_layers: int = 3,
        dropout: float = 0.2,
        head_dropout: float = 0.2,
        ffn_dim: int = 512,
        revin: bool = True,
        in_channels: int = 3,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from transformers import PatchTSTConfig, PatchTSTModel

        config = PatchTSTConfig(
            num_input_channels=in_channels,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            ffn_dim=ffn_dim,
            attention_dropout=dropout,
            ff_dropout=dropout,
            head_dropout=head_dropout,
            revin=revin,
            channel_attention=False,
        )
        self._backbone = PatchTSTModel(config)
        self._d_model = d_model
        self._pooler = PatchTokenAttentionPooling(d_model)

        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, x_signal: Tensor) -> Tensor:
        return self._pooler(self.forward_hidden(x_signal))

    def forward_hidden(self, x_signal: Tensor) -> Tensor:
        """Return PatchTST last_hidden_state for HF-style classification heads.

        Shape: (B, C, num_patches, d_model)
        """
        x = x_signal.permute(0, 2, 1)
        out = self._backbone(past_values=x)
        return out.last_hidden_state

    def load_pretrained_encoder(self, path: str) -> None:
        """Load backbone weights from a PatchTSTForPretraining checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        encoder_state = {
            k[len("model."):]: v
            for k, v in state.items()
            if k.startswith("model.")
        }
        self._backbone.load_state_dict(encoder_state, strict=False)


class NullSignalEncoder(SignalEncoder):
    """No-op encoder for TSFEL-only baselines.

    Returns a zero embedding; intended to be ignored by the model forward.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x_signal: Tensor) -> Tensor:
        batch = x_signal.shape[0]
        return x_signal.new_zeros((batch, self._output_dim))
