"""TS2Vec contrastive pretraining for CNN+LSTM-family signal encoders.

Optional self-supervised pretraining. Produces an ``encoder.*``-keyed
checkpoint that can be loaded into a supervised run via ``--init_encoder_from``
(no changes needed in main.py).

Supports encoders with ``.cnn`` and ``.lstm`` submodules — i.e.
``CNNLSTMEncoder``. PatchTST is intentionally
out of scope: HuggingFace PatchTST already has its own MAE-style pretraining
via ``--mode pretrain``.

Reference: Yue et al., "TS2Vec: Towards Universal Representation of Time
Series", AAAI 2022. https://arxiv.org/abs/2106.10466
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Hierarchical contrastive loss (vendored from ts2vec/models/losses.py for
# auditability; logic kept identical to the original implementation).
# --------------------------------------------------------------------------- #

def _instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    B = z1.size(0)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)        # 2B x T x C
    z = z.transpose(0, 1)                 # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def _temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    T = z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)        # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float = 0.5,
    temporal_unit: int = 0,
) -> torch.Tensor:
    """Multi-scale instance + temporal contrastive loss.

    z1, z2: (B, T, D) representations of two augmented views.
    """
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss = loss + alpha * _instance_contrastive_loss(z1, z2)
        if d >= temporal_unit and (1 - alpha) != 0:
            loss = loss + (1 - alpha) * _temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1 and alpha != 0:
        loss = loss + alpha * _instance_contrastive_loss(z1, z2)
        d += 1
    return loss / max(d, 1)


# --------------------------------------------------------------------------- #
# Adapter: expose per-timestep features from CNN+LSTM-family encoders.
# --------------------------------------------------------------------------- #

class _PerTimestepEncoder(nn.Module):
    """Run ``encoder.cnn`` then ``encoder.lstm`` and return all timesteps."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        if not (hasattr(encoder, "cnn") and hasattr(encoder, "lstm")):
            raise TypeError(
                "TS2Vec adapter requires an encoder with .cnn and .lstm "
                f"submodules; got {type(encoder).__name__}"
            )
        self.encoder = encoder

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, T_out, D)
        x = x_btc.transpose(1, 2)
        x = self.encoder.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.encoder.lstm(x)
        return lstm_out


# --------------------------------------------------------------------------- #
# Augmentations. Two independent views of the same window.
# --------------------------------------------------------------------------- #

def _augment(
    x: torch.Tensor,
    jitter_std: float = 0.05,
    scale_lo: float = 0.8,
    scale_hi: float = 1.2,
    mask_prob: float = 0.1,
) -> torch.Tensor:
    """Jitter + per-channel scaling + random timestep masking. x: (B, T, C)."""
    B, T, C = x.shape
    x = x + torch.randn_like(x) * jitter_std
    scale = torch.empty(B, 1, C, device=x.device).uniform_(scale_lo, scale_hi)
    x = x * scale
    keep = (torch.rand(B, T, 1, device=x.device) > mask_prob).to(x.dtype)
    return x * keep


# --------------------------------------------------------------------------- #
# Training entry point.
# --------------------------------------------------------------------------- #

def ts2vec_pretrain_encoder(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temporal_unit: int = 0,
    output_dir: str | Path | None = None,
    log_every: int = 1,
    snapshot_epochs: tuple[int, ...] = (20, 50, 100),
) -> dict:
    """Pretrain ``encoder`` (in-place) with TS2Vec hierarchical contrastive loss.

    Parameters
    ----------
    encoder
        A SignalEncoder exposing ``.cnn`` and ``.lstm`` (CNNLSTMEncoder). Trained in-place.
    dataloader
        Yields signal tensors shaped ``(B, C, T)`` — labels are not used.
    output_dir
        If given, writes ``ts2vec_best.pt`` (encoder state_dict prefixed with
        ``encoder.``, ready for ``--init_encoder_from``).

    Returns
    -------
    dict with keys ``loss_log``, ``best_loss``, ``ckpt_path``, ``snapshot_paths``.
    """
    wrapper = _PerTimestepEncoder(encoder).to(device)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    loss_log: list[float] = []
    snapshot_paths: dict[int, Path] = {}

    def _save_encoder(path: Path) -> None:
        state = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
        torch.save(state, path)

    for ep in range(epochs):
        wrapper.train()
        cum = 0.0
        n_batches = 0
        for batch in dataloader:
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            x = x.to(device)
            x_btc = x.transpose(1, 2)

            v1 = _augment(x_btc)
            v2 = _augment(x_btc)
            z1 = wrapper(v1)
            z2 = wrapper(v2)
            loss = hierarchical_contrastive_loss(z1, z2, temporal_unit=temporal_unit)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            cum += loss.item()
            n_batches += 1

        avg = cum / max(n_batches, 1)
        loss_log.append(avg)
        if (ep + 1) % log_every == 0:
            logger.info("TS2Vec ep %03d/%d | loss=%.6f", ep + 1, epochs, avg)

        if avg < best_loss:
            best_loss = avg
            if out_dir is not None:
                _save_encoder(out_dir / "ts2vec_best.pt")

        # Fixed-epoch snapshots (e.g. ep 20, 50, 100).
        if out_dir is not None and (ep + 1) in snapshot_epochs:
            snap_path = out_dir / f"ts2vec_ep{ep + 1}.pt"
            _save_encoder(snap_path)
            snapshot_paths[ep + 1] = snap_path
            logger.info("TS2Vec snapshot saved: %s", snap_path)

    logger.info("TS2Vec done. best_loss=%.6f saved=%s", best_loss, out_dir / "ts2vec_best.pt" if out_dir else None)
    return {"loss_log": loss_log, "best_loss": best_loss,
            "ckpt_path": out_dir / "ts2vec_best.pt" if out_dir else None,
            "snapshot_paths": snapshot_paths}
