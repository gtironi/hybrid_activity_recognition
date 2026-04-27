"""
Entry point for supervised, pretrain, fine-tune, and test experiments.

Usage from repository root:
  PYTHONPATH=src python -m hybrid_activity_recognition.main --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from hybrid_activity_recognition.data.dataloader import prepare_train_val_test_loaders
from hybrid_activity_recognition.models import build_hybrid_model
from hybrid_activity_recognition.training.evaluation_report import save_test_evaluation_artifacts
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
    p.add_argument("--mode", choices=("supervised", "pretrain", "finetune", "test"), required=True)
    p.add_argument(
        "--model",
        type=str,
        default="robust",
        help="Encoder name: cnn_lstm | robust | patchtst | tsfel_mlp",
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
    p.add_argument("--labeled_parquet_train", type=str, default="", help="Windowed parquet for training.")
    p.add_argument("--labeled_parquet_test", type=str, default="", help="Windowed parquet for testing.")
    p.add_argument("--labeled_parquet_val", type=str, default="", help="Optional validation parquet.")
    p.add_argument("--val_fraction", type=float, default=0.1, help="Stratified val fraction from train.")
    p.add_argument("--pretrain_parquet", type=str, default="", help="Windowed parquet for PatchTST pretraining.")

    # --- Checkpoints ---
    p.add_argument("--checkpoint", type=str, default="", help="Resume supervised training from this checkpoint.")
    p.add_argument("--patchtst_checkpoint", type=str, default="", help="Pretrained PatchTST checkpoint to load.")
    p.add_argument("--output_dir", type=str, default="experiments/runs")

    # --- Training hyperparameters ---
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=None, help="Learning rate (if omitted, uses mode default)")
    p.add_argument("--hidden_lstm", type=int, default=None)
    p.add_argument("--no_class_weights", action="store_true", help="Supervised: disable class balancing")
    p.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Supervised/finetune: freeze signal encoder (train head / TSFEL / fusion only).",
    )

    # --- PatchTST-specific ---
    p.add_argument("--context_length", type=int, default=75, help="Window length T used by PatchTST.")
    p.add_argument("--patch_len", type=int, default=12, help="Patch length (PatchTST).")
    p.add_argument("--stride", type=int, default=12, help="Stride between patches (PatchTST).")
    p.add_argument("--revin", type=int, default=1, choices=(0, 1), help="Reversible instance normalization (1=on).")
    p.add_argument("--n_layers", type=int, default=3, help="Number of Transformer layers (PatchTST).")
    p.add_argument("--n_heads", type=int, default=16, help="Number of attention heads (PatchTST).")
    p.add_argument("--d_model", type=int, default=128, help="Transformer d_model (PatchTST).")
    p.add_argument("--d_ff", type=int, default=512, help="Transformer FFN dimension (PatchTST).")
    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="PatchTST attention + feed-forward dropout (ignored for non-PatchTST models).",
    )
    p.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="PatchTST classification head dropout (HF config).",
    )
    p.add_argument(
        "--head",
        type=str,
        default="mlp",
        choices=("mlp", "linear", "patchtst_hf"),
        help="Classification head: mlp | linear | patchtst_hf (requires --model patchtst --input_mode deep_only).",
    )

    # --- Pretrain-specific ---
    p.add_argument("--pretrain_epochs", type=int, default=100)
    p.add_argument("--pretrain_lr", type=float, default=1e-3)
    p.add_argument("--mask_ratio", type=float, default=0.4, help="MAE masking ratio for PatchTST pretraining.")

    return p.parse_args()


def _build_encoder_kwargs(args) -> dict:
    """Collect encoder-specific kwargs from CLI args."""
    kwargs = {}
    if args.hidden_lstm is not None:
        kwargs["hidden_lstm"] = args.hidden_lstm
    # PatchTST kwargs (only used if encoder_name == "patchtst")
    if args.model in ("patchtst",):
        kwargs.update(
            context_length=args.context_length,
            d_model=args.d_model,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            patch_length=args.patch_len,
            patch_stride=args.stride,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            ffn_dim=args.d_ff,
            revin=bool(args.revin),
        )
        if args.patchtst_checkpoint:
            kwargs["pretrained_path"] = args.patchtst_checkpoint
    return kwargs


def _prepare_labeled_loaders(args):
    if not (args.labeled_parquet_train and args.labeled_parquet_test):
        raise SystemExit(
            "Provide --labeled_parquet_train and --labeled_parquet_test (fixed test set)."
        )
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
            context_length=args.context_length,
            patch_length=args.patch_len,
            patch_stride=args.stride,
            d_model=args.d_model,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            ffn_dim=args.d_ff,
            revin=bool(args.revin),
            mask_ratio=args.mask_ratio,
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
        head_name=args.head,
            **encoder_kwargs,
        ).to(device)
        logger.info("model (%s/%s):\n%s", args.model, args.input_mode, model)
        ckpt = args.checkpoint or str(out / "best.pt")
        trainer = Trainer(model, device, out)
        res = trainer.evaluate(test_dl, ckpt)
        metrics = classification_metrics_numpy(res["y_true"], res["y_pred"])
        paths = save_test_evaluation_artifacts(
            res["y_true"], res["y_pred"], class_names, out, stem="test"
        )
        logger.info("checkpoint=%s", ckpt)
        logger.info("test accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["f1_macro"])
        logger.info("saved confusion matrix: %s", paths["png_path"])
        logger.info("saved per-class metrics: %s", paths["json_path"])
        return

    # ---- Supervised / Finetune ----
    train_dl, val_dl, test_dl, class_names, num_classes, n_feats, _ = _prepare_labeled_loaders(args)
    logger.info("classes=%d n_tsfel_feats=%d", len(class_names), n_feats)

    encoder_kwargs = _build_encoder_kwargs(args)
    model = build_hybrid_model(
        encoder_name=args.model,
        input_mode=args.input_mode,
        num_classes=num_classes,
        n_tsfel_feats=n_feats,
        head_name=args.head,
        **encoder_kwargs,
    ).to(device)
    logger.info("model (%s/%s):\n%s", args.model, args.input_mode, model)
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
            freeze_encoder=args.freeze_encoder,
        )
        res = trainer.evaluate(test_dl, out / "best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        paths = save_test_evaluation_artifacts(
            res["y_true"], res["y_pred"], class_names, out, stem="test"
        )
        logger.info("test: acc=%.4f macro_f1=%.4f", m["accuracy"], m["f1_macro"])
        logger.info("saved confusion matrix: %s", paths["png_path"])
        logger.info("saved per-class metrics: %s", paths["json_path"])
        return

    if args.mode == "finetune":
        load_from = args.checkpoint or (out / "best.pt")
        lr = args.lr if args.lr is not None else 1e-4
        if trainer.finetune(
            train_dl,
            val_dl,
            load_path=load_from,
            epochs=args.epochs,
            lr=lr,
            freeze_encoder=args.freeze_encoder,
        ) is None:
            raise SystemExit(f"Fine-tune cancelled: checkpoint not found at {load_from}")
        res = trainer.evaluate(test_dl, out / "finetuned_best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        paths = save_test_evaluation_artifacts(
            res["y_true"], res["y_pred"], class_names, out, stem="test"
        )
        logger.info("test: acc=%.4f macro_f1=%.4f", m["accuracy"], m["f1_macro"])
        logger.info("saved confusion matrix: %s", paths["png_path"])
        logger.info("saved per-class metrics: %s", paths["json_path"])
        return


if __name__ == "__main__":
    main()
