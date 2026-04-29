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

from hybrid_activity_recognition.training.loss import log_magnitude_loss, contrastive_loss

logger = logging.getLogger(__name__)


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
        early_stopping_patience: int = 5,
        resume_from: str | Path | None = None,
    ) -> Path:
        from transformers import PatchTSTConfig, PatchTSTForPretraining
        import sys

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
        
        # 1. ADDED AMP SCALER FOR SPEED
        scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu', enabled=self.device.type == 'cuda')

        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        start_epoch = 0
        stall = 0

        ckpt_path = self.output_dir / "checkpoint.pt"
        if resume_from is not None and Path(resume_from).is_file():
            ckpt = torch.load(resume_from, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            best_state = ckpt["best_state"]
            stall = ckpt.get("stall", 0)
            logger.info("Resuming pretraining from epoch %d", start_epoch)

        # Pre-calculate total batches for the progress print
        total_batches_per_epoch = len(train_dl)

        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            
            for x_signal in train_dl:
                x = x_signal.to(self.device).permute(0, 2, 1)
                optimizer.zero_grad(set_to_none=True)
                
                # 2. WRAPPED IN AUTOCAST FOR 2X SPEED BOOST
                with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', enabled=self.device.type == 'cuda'):
                    out1 = model(past_values=x)
                    out2 = model(past_values=x)
                    
                    loss_mse = (out1.loss + out2.loss) / 2.0
                    lmm1 = log_magnitude_loss(out1.prediction_output, x)
                    lmm2 = log_magnitude_loss(out2.prediction_output, x)
                    loss_gen = loss_mse + 0.1 * (lmm1 + lmm2)
                    
                    z1 = out1.last_hidden_state.mean(dim=(1, 2))
                    z2 = out2.last_hidden_state.mean(dim=(1, 2))
                    loss_con = contrastive_loss(z1, z2)
                    
                    loss = loss_gen + 0.1 * loss_con
                
                # 3. SCALER BACKWARD PASS
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                n_batches += 1
                
                # 4. UNBUFFERED BATCH LOGGING (Prints roughly every few minutes)
                if n_batches % 500 == 0 or n_batches == total_batches_per_epoch:
                    # flush=True forces Kaggle to render the text instantly
                    print(f"Epoch {epoch + 1}/{epochs} | Batch {n_batches}/{total_batches_per_epoch} | Current Loss: {loss.item():.4f}", flush=True)

            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            
            print(f"--- Epoch {epoch + 1} Completed | Avg Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} ---", flush=True)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, self.output_dir / "best.pt")
                stall = 0
            else:
                stall += 1
                if stall >= early_stopping_patience:
                    print(f"Early stopping triggered. No improvement for {early_stopping_patience} epochs.", flush=True)
                    break

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "stall": stall,
                },
                ckpt_path,
            )

        torch.save(best_state, self.output_dir / "best.pt")
        print(f"Pretraining done. Best loss={best_loss:.6f}. Saved to {self.output_dir / 'best.pt'}", flush=True)
        return self.output_dir / "best.pt"