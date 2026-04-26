"""Load PatchTSTForPretraining checkpoints into PatchTSTForClassification (backbone only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import PatchTSTConfig, PatchTSTForClassification

_PRETRAIN_MUST_MATCH = (
    "num_input_channels",
    "context_length",
    "patch_length",
    "patch_stride",
    "d_model",
    "num_hidden_layers",
    "num_attention_heads",
    "ffn_dim",
    "use_cls_token",
)


def assert_encoder_config_compatible(
    pretrain_cfg: dict[str, Any], classification_config: PatchTSTConfig
) -> None:
    """Raise ValueError if saved pretrain config does not match classifier encoder layout."""
    for key in _PRETRAIN_MUST_MATCH:
        a, b = pretrain_cfg.get(key), getattr(classification_config, key, None)
        if a != b:
            raise ValueError(
                f"Pretrain checkpoint config mismatch on {key!r}: "
                f"ckpt={a!r} vs classifier={b!r}. Use the same architecture flags for both stages."
            )


def _checkpoint_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def load_pretrain_into_classifier(
    model: PatchTSTForClassification,
    checkpoint_path: Path | str,
    *,
    check_config: bool = True,
) -> tuple[int, list[str]]:
    """
    Copy tensors from a ``PatchTSTForPretraining`` save into ``model`` when shapes match.
    The classification ``head.*`` is usually skipped and stays randomly initialized.

    Returns:
        (num_loaded, list of skip reasons for mismatched/missing keys)
    """
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if check_config and "config" in ckpt:
        assert_encoder_config_compatible(ckpt["config"], model.config)
    src = _checkpoint_state_dict(ckpt)
    tgt = model.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in src.items():
        if k not in tgt:
            skipped.append(f"{k}: not in classifier")
            continue
        if v.shape != tgt[k].shape:
            skipped.append(f"{k}: shape {tuple(v.shape)} != {tuple(tgt[k].shape)}")
            continue
        to_load[k] = v
    model.load_state_dict(to_load, strict=False)
    return len(to_load), skipped
