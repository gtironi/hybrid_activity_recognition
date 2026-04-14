"""
Entry point for supervised, pretrain, fine-tune, FixMatch, and test experiments.

Usage from repository root:
  PYTHONPATH=src python -m hybrid_activity_recognition.main --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from hybrid_activity_recognition.data.dataloader import (
    prepare_supervised_dataloaders,
    prepare_train_val_test_loaders,
    prepare_unlabeled_dataloader,
)
from hybrid_activity_recognition.models.modular import build_hybrid_model
from hybrid_activity_recognition.training.metrics import classification_metrics_numpy
from hybrid_activity_recognition.training.trainer import Trainer
from hybrid_activity_recognition.utils.logging import setup_logging
from hybrid_activity_recognition.utils.repro import set_seed

logger = logging.getLogger(__name__)


def _ensure_src_on_path():
    here = Path(__file__).resolve()
    src_root = here.parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Activity Recognition — experiments CLI")

    # --- Mode & model ---
    p.add_argument("--mode", choices=("supervised", "pretrain", "finetune", "fixmatch", "test"), required=True)
    p.add_argument(
        "--model",
        type=str,
        default="robust",
        help=(
            "Encoder name: cnn_lstm | robust | patchtst | tsfel_mlp "
            "(or legacy: hybrid_cnn_lstm, robust_hybrid)"
        ),
    )
    p.add_argument(
        "--input_mode",
        choices=("deep_only", "hybrid", "tsfel_only"),
        default="hybrid",
        help=(
            "deep_only = encoder → head; hybrid = encoder + TSFEL → fusion → head; "
            "tsfel_only = TSFEL → head"
        ),
    )

    # --- Data ---
    p.add_argument("--labeled_parquet", type=str, default="", help="Single windowed parquet (legacy 80/10/10 split).")
    p.add_argument("--labeled_parquet_train", type=str, default="", help="Windowed parquet for training.")
    p.add_argument("--labeled_parquet_test", type=str, default="", help="Windowed parquet for testing.")
    p.add_argument("--labeled_parquet_val", type=str, default="", help="Optional validation parquet.")
    p.add_argument("--val_fraction", type=float, default=0.05, help="Stratified val fraction from train.")
    p.add_argument("--unlabeled_parquet", type=str, default="", help="Required for fixmatch mode.")
    p.add_argument("--pretrain_parquet", type=str, default="", help="Windowed parquet for PatchTST pretraining.")

    # --- Checkpoints ---
    p.add_argument("--checkpoint", type=str, default="", help="Resume supervised training from this checkpoint.")
    p.add_argument("--patchtst_checkpoint", type=str, default="", help="Pretrained PatchTST checkpoint to load.")
    p.add_argument("--output_dir", type=str, default="experiments/runs")

    # --- Training hyperparameters ---
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=None, help="Learning rate (if omitted, uses mode default)")
    p.add_argument("--hidden_lstm", type=int, default=None)
    p.add_argument("--no_class_weights", action="store_true", help="Supervised: disable class balancing")

    # --- PatchTST-specific ---
    p.add_argument("--patchtst_d_model", type=int, default=128)
    p.add_argument("--patchtst_num_layers", type=int, default=3)
    p.add_argument("--patchtst_num_heads", type=int, default=4)
    p.add_argument("--patchtst_patch_length", type=int, default=8)
    p.add_argument("--patchtst_patch_stride", type=int, default=8)
    p.add_argument("--patchtst_dropout", type=float, default=0.1)

    # --- Pretrain-specific ---
    p.add_argument("--pretrain_epochs", type=int, default=100)
    p.add_argument("--pretrain_lr", type=float, default=1e-3)
    p.add_argument("--pretrain_mask_ratio", type=float, default=0.4)

    # --- FixMatch-specific ---
    p.add_argument("--fixmatch_threshold", type=float, default=0.9)
    p.add_argument("--fixmatch_lambda", type=float, default=1.0)
    p.add_argument("--unlabeled_batch_mult", type=int, default=7)

    return p.parse_args()


def _build_encoder_kwargs(args) -> dict:
    """Collect encoder-specific kwargs from CLI args."""
    kwargs = {}
    if args.hidden_lstm is not None:
        kwargs["hidden_lstm"] = args.hidden_lstm
    # PatchTST kwargs (only used if encoder_name == "patchtst")
    if args.model in ("patchtst",):
        kwargs.update(
            context_length=75,
            d_model=args.patchtst_d_model,
            num_heads=args.patchtst_num_heads,
            num_layers=args.patchtst_num_layers,
            patch_length=args.patchtst_patch_length,
            patch_stride=args.patchtst_patch_stride,
            dropout=args.patchtst_dropout,
        )
        if args.patchtst_checkpoint:
            kwargs["pretrained_path"] = args.patchtst_checkpoint
    return kwargs


def _prepare_labeled_loaders(args):
    has_pair = bool(args.labeled_parquet_train) and bool(args.labeled_parquet_test)
    has_single = bool(args.labeled_parquet)
    if has_pair and has_single:
        raise SystemExit("Use --labeled_parquet OR --labeled_parquet_train/test, not both.")
    if has_pair:
        val_path = args.labeled_parquet_val.strip() or None
        return prepare_train_val_test_loaders(
            args.labeled_parquet_train,
            args.labeled_parquet_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=args.seed,
            val_fraction=args.val_fraction,
            parquet_val_path=val_path,
        )
    if has_single:
        return prepare_supervised_dataloaders(
            args.labeled_parquet,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=args.seed,
        )
    raise SystemExit(
        "Provide --labeled_parquet (internal split) or "
        "--labeled_parquet_train and --labeled_parquet_test (fixed test set)."
    )


def main():
    _ensure_src_on_path()
    args = parse_args()
    set_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("device=%s", device)

    # ---- Pretrain mode ----
    if args.mode == "pretrain":
        if not args.pretrain_parquet:
            raise SystemExit("pretrain mode requires --pretrain_parquet")

        from hybrid_activity_recognition.data.pretrain_dataset import prepare_pretrain_dataloader
        from hybrid_activity_recognition.training.pretrain_trainer import PretrainTrainer

        train_dl, _, _ = prepare_pretrain_dataloader(
            args.pretrain_parquet,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        trainer = PretrainTrainer(device, out)
        resume = args.checkpoint if args.checkpoint else None
        best_path = trainer.train(
            train_dl,
            context_length=75,
            patch_length=args.patchtst_patch_length,
            patch_stride=args.patchtst_patch_stride,
            d_model=args.patchtst_d_model,
            num_heads=args.patchtst_num_heads,
            num_layers=args.patchtst_num_layers,
            dropout=args.patchtst_dropout,
            mask_ratio=args.pretrain_mask_ratio,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            resume_from=resume,
        )
        logger.info("Pretraining complete. Best checkpoint: %s", best_path)
        return

    # ---- Test mode ----
    if args.mode == "test":
        train_dl, val_dl, test_dl, class_names, num_classes, n_feats, _ = _prepare_labeled_loaders(args)
        encoder_kwargs = _build_encoder_kwargs(args)
        model = build_hybrid_model(
            encoder_name=args.model,
            input_mode=args.input_mode,
            num_classes=num_classes,
            n_tsfel_feats=n_feats,
            **encoder_kwargs,
        ).to(device)
        ckpt = args.checkpoint or str(out / "best.pt")
        trainer = Trainer(model, device, out)
        res = trainer.evaluate(test_dl, ckpt)
        metrics = classification_metrics_numpy(res["y_true"], res["y_pred"])
        logger.info("checkpoint=%s", ckpt)
        logger.info("test accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["f1_macro"])
        return

    # ---- Supervised / Finetune / FixMatch ----
    train_dl, val_dl, test_dl, class_names, num_classes, n_feats, _ = _prepare_labeled_loaders(args)
    logger.info("classes=%d n_tsfel_feats=%d", len(class_names), n_feats)

    encoder_kwargs = _build_encoder_kwargs(args)
    model = build_hybrid_model(
        encoder_name=args.model,
        input_mode=args.input_mode,
        num_classes=num_classes,
        n_tsfel_feats=n_feats,
        **encoder_kwargs,
    ).to(device)
    trainer = Trainer(model, device, out)

    if args.mode == "supervised":
        lr = args.lr if args.lr is not None else 1e-3
        resume = args.checkpoint if args.checkpoint else None
        trainer.train_supervised(
            train_dl,
            val_dl,
            num_classes,
            epochs=args.epochs,
            lr=lr,
            use_class_weights=not args.no_class_weights,
            resume_from=resume,
        )
        res = trainer.evaluate(test_dl, out / "best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        logger.info("test: acc=%.4f macro_f1=%.4f", m["accuracy"], m["f1_macro"])
        return

    if args.mode == "finetune":
        load_from = args.checkpoint or (out / "best.pt")
        lr = args.lr if args.lr is not None else 1e-4
        if trainer.finetune(
            train_dl, val_dl, load_path=load_from, epochs=args.epochs, lr=lr
        ) is None:
            raise SystemExit(f"Fine-tune cancelled: checkpoint not found at {load_from}")
        res = trainer.evaluate(test_dl, out / "finetuned_best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        logger.info("test: acc=%.4f macro_f1=%.4f", m["accuracy"], m["f1_macro"])
        return

    if args.mode == "fixmatch":
        if not args.unlabeled_parquet:
            raise SystemExit("fixmatch requires --unlabeled_parquet")
        ul = prepare_unlabeled_dataloader(
            args.unlabeled_parquet,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            unlabeled_batch_multiplier=args.unlabeled_batch_mult,
        )
        warm = args.checkpoint or (out / "finetuned_best.pt")
        lr = args.lr if args.lr is not None else 2e-3
        trainer.train_fixmatch(
            train_dl,
            ul,
            val_dl,
            finetune_checkpoint=warm,
            epochs=args.epochs,
            lr=lr,
            threshold=args.fixmatch_threshold,
            lambda_u=args.fixmatch_lambda,
        )
        res = trainer.evaluate(test_dl, out / "fixmatch_best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        logger.info("test: acc=%.4f macro_f1=%.4f", m["accuracy"], m["f1_macro"])
        return


if __name__ == "__main__":
    main()
