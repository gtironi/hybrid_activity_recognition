"""Concrete SignalEncoder implementations.

Each encoder maps (B, C, T) sensor windows to (B, output_dim) embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hybrid_activity_recognition.models.modular.base import SignalEncoder


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
    """3 Conv1D blocks + 1-layer BiLSTM, h_n concatenation.

    Migrated from ``layers.signal_branch.RobustCNNLSTMSignalBranch``.
    Default output_dim = 2 * hidden_lstm = 256.
    Applies Kaiming initialization to all Conv1d and Linear layers.
    """

    def __init__(self, in_channels: int = 3, hidden_lstm: int = 128):
        super().__init__()
        self._output_dim = hidden_lstm * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_lstm,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
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
        _, (h_n, _) = self.lstm(x)
        return torch.cat((h_n[-2], h_n[-1]), dim=1)


class PatchTSTEncoder(SignalEncoder):
    """Wrapper around HuggingFace ``PatchTSTModel`` as a SignalEncoder.

    Transposes input from ``(B, C, T)`` to ``(B, T, C)`` internally (HF expects
    channels-last).  Mean-pools over the patch dimension to produce a fixed-size
    ``(B, d_model)`` embedding.

    If ``pretrained_path`` is given, loads encoder weights saved from a
    ``PatchTSTForPretraining`` checkpoint (strips the ``"model."`` prefix).
    """

    def __init__(
        self,
        context_length: int = 75,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
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
            dropout=dropout,
            channel_attention=False,
        )
        self._backbone = PatchTSTModel(config)
        self._d_model = d_model

        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, x_signal: Tensor) -> Tensor:
        # (B, C, T) -> (B, T, C) — HuggingFace expects channels-last
        x = x_signal.permute(0, 2, 1)
        out = self._backbone(past_values=x)
        # last_hidden_state: (B, num_channels, num_patches, d_model)
        # Mean pool over channels and patches -> (B, d_model)
        return out.last_hidden_state.mean(dim=(1, 2))

    def load_pretrained_encoder(self, path: str) -> None:
        """Load backbone weights from a PatchTSTForPretraining checkpoint."""
        state = torch.load(path, map_location="cpu")
        encoder_state = {
            k[len("model."):]: v
            for k, v in state.items()
            if k.startswith("model.")
        }
        self._backbone.load_state_dict(encoder_state, strict=False)
