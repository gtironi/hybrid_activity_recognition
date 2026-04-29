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

class EMAVectorQuantizer(nn.Module):
    """Exponential Moving Average (EMA) Vector Quantizer.
    
    Updates the discrete codebook internally during the forward pass, 
    bypassing the need for custom commitment losses in the training loop.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, N, D)
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Calculate distances between inputs and codebook vectors
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embed**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embed.t()))
        
        # Encode (find closest motion primitive)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embed).view(x.shape)
        
        # Update codebook via EMA (only during training)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(torch.matmul(encodings.t(), flat_x), alpha=1 - self.decay)
            
            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            
        # Straight-through estimator: copy gradients from quantized back to x
        quantized = x + (quantized - x).detach()
        return quantized


class MoPFormerEncoder(SignalEncoder):
    """Motion-Primitive Transformer (MoPFormer) with Vector Quantization.
    
    Translates continuous sensor streams into discrete motion primitives 
    (a vocabulary of fundamental physical movements) before transformer processing.
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        d_model: int = 128, 
        num_primitives: int = 256,
        patch_length: int = 8,
        patch_stride: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self._output_dim = d_model
        
        # 1. Patchification (Continuous)
        self.patcher = nn.Conv1d(in_channels, d_model, kernel_size=patch_length, stride=patch_stride)
        
        # 2. Discrete Tokenization
        self.quantizer = EMAVectorQuantizer(num_embeddings=num_primitives, embedding_dim=d_model)
        
        # 3. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
        
    def forward(self, x_signal: Tensor) -> Tensor:
        # Patchify: (B, C, T) -> (B, d_model, num_patches)
        patches = self.patcher(x_signal)
        
        # Permute for Quantizer & Transformer: (B, num_patches, d_model)
        patches = patches.permute(0, 2, 1)
        
        # Quantize into discrete motion primitives
        quantized_patches = self.quantizer(patches)
        
        # Process sequential grammar of primitives
        encoded = self.transformer(quantized_patches)
        
        # Mean pool over time/patches -> (B, d_model)
        return encoded.mean(dim=1)