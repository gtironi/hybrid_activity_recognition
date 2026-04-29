"""HuggingFace PatchTST encoder wrapper. No TSFEL dependencies."""

from __future__ import annotations
import torch
from torch import Tensor
from pretrain_ablations.encoders.base import Encoder


def _best_patch_length(T: int, requested: int) -> int:
    pl = min(requested, T)
    while pl > 1 and T % pl != 0:
        pl -= 1
    return pl


class PatchTSTEncoder(Encoder):
    """Wraps HF PatchTSTModel. Permutes (B,C,T)→(B,T,C) internally.
    Output: last_hidden_state.mean(dim=(1,2)) → (B, d_model).
    """

    def __init__(
        self,
        context_length: int,
        in_channels: int = 1,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from transformers import PatchTSTConfig, PatchTSTModel

        pl = _best_patch_length(context_length, patch_length)
        ps = min(patch_stride, pl)

        config = PatchTSTConfig(
            num_input_channels=in_channels,
            context_length=context_length,
            patch_length=pl,
            patch_stride=ps,
            d_model=d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            dropout=dropout,
            channel_attention=False,
        )
        self._backbone = PatchTSTModel(config)
        self._d = d_model

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            sd = state.get("model_state_dict", state)
            enc_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            if enc_sd:
                self._backbone.load_state_dict(enc_sd, strict=False)

    @property
    def output_dim(self) -> int:
        return self._d

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T) → (B, T, C) for HF
        out = self._backbone(past_values=x.permute(0, 2, 1))
        # last_hidden_state: (B, C, num_patches, d_model)
        return out.last_hidden_state.mean(dim=(1, 2))
