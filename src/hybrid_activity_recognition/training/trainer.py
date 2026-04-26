from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybrid_activity_recognition.training.loss import balanced_class_weights, supervised_loss_fn

logger = logging.getLogger(__name__)


class Trainer:
    """Supervised training, fine-tuning, and val/test evaluation loops."""

    def __init__(self, model: nn.Module, device: torch.device, output_dir: str | Path):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_supervised(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        num_classes: int,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        use_class_weights: bool = True,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.3,
        early_stopping_patience: int = 10,
        grad_clip: float = 1.0,
        checkpoint_name: str = "best.pt",
        resume_from: str | Path | None = None,
    ) -> nn.Module:
        labels = train_dl.dataset.labels.cpu().numpy()
        cw = None
        if use_class_weights:
            cw = balanced_class_weights(labels, num_classes).to(self.device)
        criterion = supervised_loss_fn(cw)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )

        best_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        stall = 0
        start_epoch = 0
        ckpt_path = self.output_dir / checkpoint_name

        # Resume from checkpoint
        if resume_from is not None and Path(resume_from).is_file():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_acc = ckpt["best_acc"]
            best_wts = ckpt["best_wts"]
            stall = ckpt["stall"]
            start_epoch = ckpt["epoch"] + 1
            logger.info("Resuming training from epoch %d (best_acc=%.2f%%)", start_epoch, best_acc)

        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for x_sig, x_feat, y in train_dl:
                x_sig, x_feat, y = x_sig.to(self.device), x_feat.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x_sig, x_feat)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                train_loss += loss.item() * x_sig.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            avg_train_loss = train_loss / len(train_dl.dataset)
            train_acc = 100.0 * correct / total

            self.model.eval()
            val_loss = 0.0
            v_correct = v_total = 0
            with torch.no_grad():
                for x_sig, x_feat, y in val_dl:
                    x_sig, x_feat, y = x_sig.to(self.device), x_feat.to(self.device), y.to(self.device)
                    logits = self.model(x_sig, x_feat)
                    val_loss += criterion(logits, y).item() * x_sig.size(0)
                    v_correct += (logits.argmax(1) == y).sum().item()
                    v_total += y.size(0)
            avg_val_loss = val_loss / len(val_dl.dataset)
            val_acc = 100.0 * v_correct / v_total

            logger.info(
                "Ep %03d/%d | train_loss=%.4f acc=%.2f%% | val_loss=%.4f val_acc=%.2f%%",
                epoch + 1, epochs, avg_train_loss, train_acc, avg_val_loss, val_acc,
            )
            scheduler.step(avg_val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                best_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), ckpt_path)
                stall = 0
            else:
                stall += 1
                if stall >= early_stopping_patience:
                    logger.info("Early stopping (no improvement for %d epochs).", early_stopping_patience)
                    break

            # Periodic checkpoint for resume
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "best_wts": best_wts,
                    "stall": stall,
                },
                self.output_dir / "checkpoint.pt",
            )

        self.model.load_state_dict(best_wts)
        return self.model

    def finetune(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        load_path: str | Path,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        grad_clip: float = 1.0,
        checkpoint_name: str = "finetuned_best.pt",
    ) -> nn.Module | None:
        load_path = Path(load_path)
        if not load_path.is_file():
            logger.warning("Checkpoint not found: %s", load_path)
            return None
        self.model.load_state_dict(torch.load(load_path, map_location=self.device, weights_only=True))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )
        best_acc = 0.0
        ckpt_path = self.output_dir / checkpoint_name

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct = total = 0
            for x_sig, x_feat, y in train_dl:
                x_sig, x_feat, y = x_sig.to(self.device), x_feat.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x_sig, x_feat)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                train_loss += loss.item() * x_sig.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            avg_train_loss = train_loss / len(train_dl.dataset)
            train_acc = 100.0 * correct / total

            self.model.eval()
            val_loss = 0.0
            v_correct = v_total = 0
            with torch.no_grad():
                for x_sig, x_feat, y in val_dl:
                    x_sig, x_feat, y = x_sig.to(self.device), x_feat.to(self.device), y.to(self.device)
                    logits = self.model(x_sig, x_feat)
                    val_loss += criterion(logits, y).item() * x_sig.size(0)
                    v_correct += (logits.argmax(1) == y).sum().item()
                    v_total += y.size(0)
            avg_val_loss = val_loss / len(val_dl.dataset)
            val_acc = 100.0 * v_correct / v_total
            logger.info(
                "Finetune Ep %03d/%d | train_loss=%.4f acc=%.2f%% | val_loss=%.4f val_acc=%.2f%%",
                epoch + 1, epochs, avg_train_loss, train_acc, avg_val_loss, val_acc,
            )
            scheduler.step(avg_val_loss)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), ckpt_path)

        return self.model

    def evaluate(self, data_loader: DataLoader, checkpoint: str | Path | None = None) -> dict:
        if checkpoint and Path(checkpoint).is_file():
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for x_sig, x_feat, y in data_loader:
                x_sig, x_feat = x_sig.to(self.device), x_feat.to(self.device)
                logits = self.model(x_sig, x_feat)
                pred = logits.argmax(1).cpu().numpy()
                ys.append(y.numpy())
                preds.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        acc = float((y_true == y_pred).mean())
        return {"accuracy": acc, "y_true": y_true, "y_pred": y_pred}
