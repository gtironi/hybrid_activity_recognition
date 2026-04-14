"""Self-supervised pretraining for PatchTST using masked auto-encoding (MAE).

Uses HuggingFace ``PatchTSTForPretraining`` which handles masking internally.
Saves periodic checkpoints for resume and a ``best.pt`` with the lowest loss.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """MAE pretraining loop for PatchTST.

    Parameters
    ----------
    device : torch.device
        Target device.
    output_dir : str | Path
        Directory where checkpoints are saved.
    """

    def __init__(self, device: torch.device, output_dir: str | Path):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_dl: DataLoader,
        context_length: int = 75,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        mask_ratio: float = 0.4,
        in_channels: int = 3,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        resume_from: str | Path | None = None,
    ) -> Path:
        """Run MAE pretraining and return path to best checkpoint.

        Returns
        -------
        Path
            Path to ``best.pt`` (best loss, model state_dict only).
        """
        from transformers import PatchTSTConfig, PatchTSTForPretraining

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
            do_mask_input=True,
            random_mask_ratio=mask_ratio,
        )
        model = PatchTSTForPretraining(config).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        start_epoch = 0

        # Resume
        ckpt_path = self.output_dir / "checkpoint.pt"
        if resume_from is not None and Path(resume_from).is_file():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            best_state = ckpt["best_state"]
            logger.info("Resuming pretraining from epoch %d", start_epoch)

        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for x_signal in train_dl:
                # x_signal: (B, C, T) -> (B, T, C) for HF
                x = x_signal.to(self.device).permute(0, 2, 1)
                optimizer.zero_grad(set_to_none=True)
                out = model(past_values=x)
                loss = out.loss
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            logger.info(
                "Pretrain Ep %03d/%d | loss=%.6f | lr=%.2e",
                epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0],
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, self.output_dir / "best.pt")

            # Periodic checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                },
                ckpt_path,
            )

        # Final save
        torch.save(best_state, self.output_dir / "best.pt")
        logger.info("Pretraining done. Best loss=%.6f. Saved to %s", best_loss, self.output_dir / "best.pt")
        return self.output_dir / "best.pt"
