from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybrid_activity_recognition.training.ensemble import (
    align_rf_proba_to_class_indices,
    combine_and_predict,
    ensemble_metrics,
    fit_stacking_logistic,
    rf_proba_from_features,
)
from hybrid_activity_recognition.training.grl_schedule import ganin_grl_lambda
from hybrid_activity_recognition.training.loss import build_supervised_criterion
from hybrid_activity_recognition.training.metrics import classification_metrics_numpy

logger = logging.getLogger(__name__)


def _iter_trainable_params(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


def _unpack_batch(batch):
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    return batch[0], batch[1], batch[2], None


def _behaviour_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


class Trainer:
    """Supervised training, fine-tuning, and val/test evaluation loops."""

    def __init__(self, model: nn.Module, device: torch.device, output_dir: str | Path):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _apply_signal_encoder_freeze(self, freeze: bool) -> None:
        enc = getattr(self.model, "encoder", None)
        if enc is None or type(enc).__name__ == "NullSignalEncoder":
            return
        for p in enc.parameters():
            p.requires_grad = not freeze
        logger.info("Signal encoder requires_grad=%s", not freeze)

    @torch.no_grad()
    def _evaluate_split(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> dict:
        self.model.eval()
        val_loss = 0.0
        ys, preds = [], []
        n = len(data_loader.dataset)
        for batch in data_loader:
            x_sig, x_feat, y, _ = _unpack_batch(batch)
            x_sig = x_sig.to(self.device)
            x_feat = x_feat.to(self.device)
            y = y.to(self.device)
            logits = _behaviour_logits(self.model(x_sig, x_feat))
            val_loss += criterion(logits, y).item() * x_sig.size(0)
            pred = logits.argmax(1)
            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        metrics = classification_metrics_numpy(y_true, y_pred)
        return {
            "loss": val_loss / n,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def train_supervised(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        num_classes: int,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        use_class_weights: bool = True,
        loss_criterion: str = "weighted_ce",
        focal_gamma: float = 2.0,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.3,
        early_stopping_patience: int = 25,
        grad_clip: float = 1.0,
        checkpoint_name: str = "best.pt",
        resume_from: str | Path | None = None,
        freeze_encoder: bool = False,
        adversarial_subject_alignment: bool = False,
        adversarial_beta: float = 0.1,
    ) -> nn.Module:
        """Stage 2: checkpoint selected by validation macro-F1 (minority-aware)."""
        best_wts = copy.deepcopy(self.model.state_dict())
        best_f1 = -1.0
        best_acc = 0.0
        stall = 0
        start_epoch = 0
        ckpt_path = self.output_dir / checkpoint_name
        resume_ckpt = None

        if resume_from is not None and Path(resume_from).is_file():
            resume_ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            self.model.load_state_dict(resume_ckpt["model_state_dict"])
            best_f1 = float(resume_ckpt.get("best_f1_macro", resume_ckpt.get("best_acc", 0.0)))
            best_acc = float(resume_ckpt.get("best_acc", 0.0))
            best_wts = resume_ckpt["best_wts"]
            stall = resume_ckpt["stall"]
            start_epoch = resume_ckpt["epoch"] + 1
            logger.info(
                "Resuming from epoch %d (best_val_f1_macro=%.4f best_val_acc=%.2f%%)",
                start_epoch,
                best_f1,
                best_acc * 100.0,
            )

        self._apply_signal_encoder_freeze(freeze_encoder)

        labels = train_dl.dataset.labels.cpu().numpy()
        criterion = build_supervised_criterion(
            loss_criterion,
            labels,
            num_classes,
            self.device,
            use_class_weights=use_class_weights,
            focal_gamma=focal_gamma,
        )
        use_adversarial = (
            adversarial_subject_alignment
            and getattr(self.model, "subject_discriminator", None) is not None
        )
        if adversarial_subject_alignment and not use_adversarial:
            logger.warning(
                "adversarial_subject_alignment requested but model has no subject_discriminator; disabled."
            )
        subject_criterion = nn.CrossEntropyLoss() if use_adversarial else None
        logger.info(
            "loss_criterion=%s class_weights=%s adversarial=%s beta=%s",
            loss_criterion,
            use_class_weights,
            use_adversarial,
            adversarial_beta if use_adversarial else "n/a",
        )

        optimizer = torch.optim.AdamW(_iter_trainable_params(self.model), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )

        if resume_ckpt is not None and not freeze_encoder and "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not load optimizer/scheduler state (%s); starting fresh.", e)

        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_loss = 0.0
            correct = total = 0
            grl_alpha = ganin_grl_lambda(epoch, epochs) if use_adversarial else 0.0

            for batch in train_dl:
                x_sig, x_feat, y, subj = _unpack_batch(batch)
                x_sig = x_sig.to(self.device)
                x_feat = x_feat.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                if use_adversarial:
                    if subj is None:
                        raise RuntimeError("Adversarial training requires subject_idx in train batches.")
                    subj = subj.to(self.device)
                    behaviour_logits, subject_logits = self.model(
                        x_sig,
                        x_feat,
                        compute_adversarial=True,
                        grl_alpha=grl_alpha,
                    )
                    loss_beh = criterion(behaviour_logits, y)
                    loss_sub = subject_criterion(subject_logits, subj)
                    loss = loss_beh + adversarial_beta * loss_sub
                    logits = behaviour_logits
                else:
                    logits = _behaviour_logits(self.model(x_sig, x_feat))
                    loss = criterion(logits, y)

                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(_iter_trainable_params(self.model)), grad_clip
                    )
                optimizer.step()
                train_loss += loss.item() * x_sig.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            avg_train_loss = train_loss / len(train_dl.dataset)
            train_acc = 100.0 * correct / total

            val_out = self._evaluate_split(val_dl, criterion)
            val_acc_pct = val_out["accuracy"] * 100.0
            val_f1 = val_out["f1_macro"]

            if use_adversarial:
                logger.info(
                    "Ep %03d/%d | grl_alpha=%.3f train_loss=%.4f acc=%.2f%% | "
                    "val_loss=%.4f val_acc=%.2f%% val_f1_macro=%.4f",
                    epoch + 1,
                    epochs,
                    grl_alpha,
                    avg_train_loss,
                    train_acc,
                    val_out["loss"],
                    val_acc_pct,
                    val_f1,
                )
            else:
                logger.info(
                    "Ep %03d/%d | train_loss=%.4f acc=%.2f%% | val_loss=%.4f val_acc=%.2f%% val_f1_macro=%.4f",
                    epoch + 1,
                    epochs,
                    avg_train_loss,
                    train_acc,
                    val_out["loss"],
                    val_acc_pct,
                    val_f1,
                )
            scheduler.step(val_out["loss"])

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_out["accuracy"]
                best_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), ckpt_path)
                stall = 0
            else:
                stall += 1
                if stall >= early_stopping_patience:
                    logger.info(
                        "Early stopping on val macro-F1 (no improvement for %d epochs).",
                        early_stopping_patience,
                    )
                    break

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_f1_macro": best_f1,
                    "best_acc": best_acc,
                    "best_wts": best_wts,
                    "stall": stall,
                },
                self.output_dir / "checkpoint.pt",
            )

        self.model.load_state_dict(best_wts)
        logger.info("Loaded best checkpoint (val_f1_macro=%.4f val_acc=%.2f%%)", best_f1, best_acc * 100.0)
        return self.model

    def finetune(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        load_path: str | Path,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        early_stopping_patience: int = 15,
        grad_clip: float = 1.0,
        checkpoint_name: str = "finetuned_best.pt",
        freeze_encoder: bool = False,
    ) -> nn.Module | None:
        """Stage 3: plain CE + label smoothing; checkpoint on validation accuracy."""
        load_path = Path(load_path)
        if not load_path.is_file():
            logger.warning("Checkpoint not found: %s", load_path)
            return None
        self.model.load_state_dict(torch.load(load_path, map_location=self.device, weights_only=True), strict=False)
        self._apply_signal_encoder_freeze(freeze_encoder)

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(_iter_trainable_params(self.model), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )
        best_acc = 0.0
        best_wts = copy.deepcopy(self.model.state_dict())
        stall = 0
        ckpt_path = self.output_dir / checkpoint_name

        logger.info(
            "Finetune: lr=%.2e label_smoothing=%.3f early_stopping_patience=%d",
            lr,
            label_smoothing,
            early_stopping_patience,
        )

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct = total = 0
            for batch in train_dl:
                x_sig, x_feat, y, _ = _unpack_batch(batch)
                x_sig = x_sig.to(self.device)
                x_feat = x_feat.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = _behaviour_logits(self.model(x_sig, x_feat))
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(_iter_trainable_params(self.model)), grad_clip
                    )
                optimizer.step()
                train_loss += loss.item() * x_sig.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            avg_train_loss = train_loss / len(train_dl.dataset)
            train_acc = 100.0 * correct / total

            val_out = self._evaluate_split(val_dl, criterion)
            val_acc = val_out["accuracy"]
            val_acc_pct = val_acc * 100.0

            logger.info(
                "Finetune Ep %03d/%d | train_loss=%.4f acc=%.2f%% | val_loss=%.4f val_acc=%.2f%% val_f1_macro=%.4f",
                epoch + 1,
                epochs,
                avg_train_loss,
                train_acc,
                val_out["loss"],
                val_acc_pct,
                val_out["f1_macro"],
            )
            scheduler.step(val_out["loss"])

            if val_acc > best_acc:
                best_acc = val_acc
                best_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), ckpt_path)
                stall = 0
            else:
                stall += 1
                if stall >= early_stopping_patience:
                    logger.info(
                        "Finetune early stopping on val accuracy (no improvement for %d epochs).",
                        early_stopping_patience,
                    )
                    break

        self.model.load_state_dict(best_wts)
        logger.info("Finetune complete (best_val_acc=%.2f%%)", best_acc * 100.0)
        return self.model

    @torch.no_grad()
    def predict_proba(
        self,
        data_loader: DataLoader,
        checkpoint: str | Path | None = None,
        num_classes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_true, proba) with shape (N,) and (N, C)."""
        if checkpoint and Path(checkpoint).is_file():
            self.model.load_state_dict(
                torch.load(checkpoint, map_location=self.device, weights_only=True)
            )
        self.model.eval()
        ys, probas = [], []
        for batch in data_loader:
            x_sig, x_feat, y, _ = _unpack_batch(batch)
            x_sig = x_sig.to(self.device)
            x_feat = x_feat.to(self.device)
            logits = _behaviour_logits(self.model(x_sig, x_feat))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            ys.append(y.numpy())
            probas.append(proba)
        y_true = np.concatenate(ys)
        proba_all = np.concatenate(probas)
        if num_classes is not None and proba_all.shape[1] != num_classes:
            raise ValueError(
                f"Expected {num_classes} classes, got proba shape {proba_all.shape}"
            )
        return y_true, proba_all

    def rf_proba_from_dataset(
        self, rf_artifacts: dict, dataset, num_classes: int
    ) -> np.ndarray:
        features = dataset.features.cpu().numpy()
        proba = rf_proba_from_features(rf_artifacts, features)
        clf = rf_artifacts["classifier"]
        return align_rf_proba_to_class_indices(proba, clf.classes_, num_classes)

    def run_ensemble_evaluation(
        self,
        *,
        val_dl: DataLoader,
        test_dl: DataLoader,
        rf_artifacts: dict,
        deep_checkpoint: str | Path,
        num_classes: int,
        method: str = "soft_vote",
        weight_deep: float = 0.5,
    ) -> dict:
        """Late fusion: deep softmax + TSFEL Random Forest on val (fit) and test (eval)."""
        ckpt = Path(deep_checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Deep checkpoint not found: {ckpt}")

        y_val, proba_deep_val = self.predict_proba(val_dl, ckpt, num_classes=num_classes)
        proba_rf_val = self.rf_proba_from_dataset(rf_artifacts, val_dl.dataset, num_classes)

        y_test, proba_deep_test = self.predict_proba(test_dl, ckpt, num_classes=num_classes)
        proba_rf_test = self.rf_proba_from_dataset(rf_artifacts, test_dl.dataset, num_classes)

        meta = None
        if method == "stacking":
            meta = fit_stacking_logistic(proba_deep_val, proba_rf_val, y_val)
            logger.info("Fitted logistic stacking meta-learner on validation split.")

        _, pred_val = combine_and_predict(
            proba_deep_val,
            proba_rf_val,
            y_val,
            method=method,  # type: ignore[arg-type]
            weight_deep=weight_deep,
            meta=meta,
        )
        proba_test, pred_test = combine_and_predict(
            proba_deep_test,
            proba_rf_test,
            y_test,
            method=method,  # type: ignore[arg-type]
            weight_deep=weight_deep,
            meta=meta,
        )

        val_m = ensemble_metrics(y_val, pred_val)
        test_m = ensemble_metrics(y_test, pred_test)
        logger.info(
            "Ensemble val: acc=%.4f f1_macro=%.4f | test: acc=%.4f f1_macro=%.4f",
            val_m["accuracy"],
            val_m["f1_macro"],
            test_m["accuracy"],
            test_m["f1_macro"],
        )
        return {
            "val": val_m,
            "test": test_m,
            "y_true": y_test,
            "y_pred": pred_test,
            "proba": proba_test,
        }

    def evaluate(self, data_loader: DataLoader, checkpoint: str | Path | None = None) -> dict:
        if checkpoint and Path(checkpoint).is_file():
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for batch in data_loader:
                x_sig, x_feat, y, _ = _unpack_batch(batch)
                x_sig = x_sig.to(self.device)
                x_feat = x_feat.to(self.device)
                logits = _behaviour_logits(self.model(x_sig, x_feat))
                pred = logits.argmax(1).cpu().numpy()
                ys.append(y.numpy())
                preds.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        acc = float((y_true == y_pred).mean())
        return {"accuracy": acc, "y_true": y_true, "y_pred": y_pred}
