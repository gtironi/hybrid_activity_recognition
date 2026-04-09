from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybrid_activity_recognition.training.augment import SensorFixMatchAugment
from hybrid_activity_recognition.training.loss import balanced_class_weights, supervised_loss_fn
def _cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer:
    """Loops de treino supervisionado, fine-tune e FixMatch; avaliação no val/test."""

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
        early_stopping_patience: int = 25,
        grad_clip: float = 1.0,
        checkpoint_name: str = "supervised_best.pt",
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
        ckpt_path = self.output_dir / checkpoint_name

        for epoch in range(epochs):
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

            print(
                f"Ep {epoch + 1:03d}/{epochs} | train_loss={avg_train_loss:.4f} acc={train_acc:.2f}% | "
                f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.2f}%"
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
                    print(f"Early stopping (sem melhoria por {early_stopping_patience} épocas).")
                    break

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
            print(f"Checkpoint não encontrado: {load_path}")
            return None
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))

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
            print(
                f"Finetune Ep {epoch + 1:03d}/{epochs} | train_loss={avg_train_loss:.4f} acc={train_acc:.2f}% | "
                f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.2f}%"
            )
            scheduler.step(avg_val_loss)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), ckpt_path)

        return self.model

    def train_fixmatch(
        self,
        labeled_dl: DataLoader,
        unlabeled_dl: DataLoader,
        val_dl: DataLoader,
        finetune_checkpoint: str | Path | None,
        epochs: int = 20,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        threshold: float = 0.9,
        lambda_u: float = 1.0,
        grad_clip: float = 1.0,
        checkpoint_name: str = "fixmatch_best.pt",
    ) -> nn.Module:
        if finetune_checkpoint and Path(finetune_checkpoint).is_file():
            self.model.load_state_dict(torch.load(finetune_checkpoint, map_location=self.device))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        augmenter = SensorFixMatchAugment(self.device)
        unlabeled_iter = _cycle(unlabeled_dl)
        best_acc = 0.0
        ckpt_path = self.output_dir / checkpoint_name

        for epoch in range(epochs):
            self.model.train()
            totals = {"loss": 0.0, "s": 0.0, "u": 0.0, "mask": 0.0, "steps": 0}

            for x_s_lab, x_f_lab, y_lab in labeled_dl:
                x_s_lab = x_s_lab.to(self.device)
                x_f_lab = x_f_lab.to(self.device)
                y_lab = y_lab.to(self.device)

                x_s_u, x_f_u = next(unlabeled_iter)
                x_s_u = x_s_u.to(self.device)
                x_f_u = x_f_u.to(self.device)
                bs = x_s_lab.size(0)
                x_s_u, x_f_u = x_s_u[:bs], x_f_u[:bs]

                x_s_aug, x_f_aug = augmenter.weak_aug(x_s_lab, x_f_lab)
                logits_lab = self.model(x_s_aug, x_f_aug)
                loss_s = torch.nn.functional.cross_entropy(logits_lab, y_lab)

                with torch.no_grad():
                    w_s, w_f = augmenter.weak_aug(x_s_u, x_f_u)
                    logits_weak = self.model(w_s, w_f)
                    probs_weak = torch.softmax(logits_weak, dim=1)
                    max_probs, pseudo = torch.max(probs_weak, dim=1)
                    mask = max_probs.ge(threshold).float()
                    totals["mask"] += mask.mean().item()

                st_s, st_f = augmenter.strong_aug(x_s_u, x_f_u)
                logits_strong = self.model(st_s, st_f)
                loss_u_elem = torch.nn.functional.cross_entropy(
                    logits_strong, pseudo, reduction="none"
                )
                loss_u = (loss_u_elem * mask).mean()
                loss = loss_s + lambda_u * loss_u

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()

                totals["loss"] += loss.item()
                totals["s"] += loss_s.item()
                totals["u"] += loss_u.item()
                totals["steps"] += 1

            self.model.eval()
            v_ok = v_n = 0
            with torch.no_grad():
                for x_s, x_f, y in val_dl:
                    x_s, x_f, y = x_s.to(self.device), x_f.to(self.device), y.to(self.device)
                    pred = self.model(x_s, x_f).argmax(1)
                    v_ok += (pred == y).sum().item()
                    v_n += y.size(0)
            val_acc = 100.0 * v_ok / max(1, v_n)
            steps = max(1, totals["steps"])
            print(
                f"FixMatch Ep {epoch + 1:03d}/{epochs} | loss={totals['loss'] / steps:.4f} "
                f"(sup={totals['s'] / steps:.4f} unsup={totals['u'] / steps:.4f}) "
                f"| mask~{totals['mask'] / steps:.2%} | val_acc={val_acc:.2f}%"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), ckpt_path)

        return self.model

    def evaluate(self, data_loader: DataLoader, checkpoint: str | Path | None = None) -> dict:
        if checkpoint and Path(checkpoint).is_file():
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
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
