"""
Ponto de entrada: experimentos supervisionados, fine-tune, FixMatch e teste.

Uso típico a partir da raiz do repositório:
  PYTHONPATH=src python -m hybrid_activity_recognition.main --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from hybrid_activity_recognition.data.dataloader import (
    prepare_supervised_dataloaders,
    prepare_train_val_test_loaders,
    prepare_unlabeled_dataloader,
)
from hybrid_activity_recognition.models.hybrid_cnn_lstm.model import HybridCNNLSTM
from hybrid_activity_recognition.models.robust_hybrid.model import RobustHybridModel
from hybrid_activity_recognition.training.metrics import classification_metrics_numpy
from hybrid_activity_recognition.training.trainer import Trainer
from hybrid_activity_recognition.utils.repro import set_seed


def _ensure_src_on_path():
    here = Path(__file__).resolve()
    src_root = here.parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def build_model(name: str, num_classes: int, n_feats: int, hidden_lstm: int | None):
    if name == "hybrid_cnn_lstm":
        h = hidden_lstm if hidden_lstm is not None else 64
        return HybridCNNLSTM(num_classes=num_classes, n_features_tsfel=n_feats, hidden_lstm=h)
    if name == "robust_hybrid":
        h = hidden_lstm if hidden_lstm is not None else 128
        return RobustHybridModel(num_classes=num_classes, n_features_tsfel=n_feats, hidden_lstm=h)
    raise ValueError(f"Modelo desconhecido: {name}. Use hybrid_cnn_lstm ou robust_hybrid.")


def parse_args():
    p = argparse.ArgumentParser(description="Treino híbrido AcTBeCalf (PyTorch)")
    p.add_argument("--mode", choices=("supervised", "finetune", "fixmatch", "test"), required=True)
    p.add_argument("--model", choices=("hybrid_cnn_lstm", "robust_hybrid"), default="robust_hybrid")
    p.add_argument(
        "--labeled_parquet",
        type=str,
        default="",
        help="Um único Parquet janelado; split interno 80/10/10 (legado).",
    )
    p.add_argument(
        "--labeled_parquet_train",
        type=str,
        default="",
        help="Parquet janelado de treino (usa com --labeled_parquet_test; sem vazamento de normalização).",
    )
    p.add_argument("--labeled_parquet_test", type=str, default="", help="Parquet janelado de teste fixo.")
    p.add_argument(
        "--labeled_parquet_val",
        type=str,
        default="",
        help="Parquet janelado de validação opcional (senão usa --val_fraction no treino).",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fração estratificada do treino para validação (ignorada se --labeled_parquet_val).",
    )
    p.add_argument("--unlabeled_parquet", type=str, default="", help="Obrigatório se mode=fixmatch")
    p.add_argument("--output_dir", type=str, default="experiments/runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", help="cuda ou cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=None, help="Se omitido, usa default do modo")
    p.add_argument("--hidden_lstm", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default="", help="Checkpoint para test/finetune/fixmatch warm-start")
    p.add_argument("--no_class_weights", action="store_true", help="Supervised: sem pesos por classe")
    p.add_argument("--fixmatch_threshold", type=float, default=0.9)
    p.add_argument("--fixmatch_lambda", type=float, default=1.0)
    p.add_argument("--unlabeled_batch_mult", type=int, default=7)
    return p.parse_args()


def _prepare_labeled_loaders(args):
    has_pair = bool(args.labeled_parquet_train) and bool(args.labeled_parquet_test)
    has_single = bool(args.labeled_parquet)
    if has_pair and has_single:
        raise SystemExit("Use apenas --labeled_parquet OU o par train/test, não ambos.")
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
        "Indique --labeled_parquet (split interno) ou "
        "--labeled_parquet_train e --labeled_parquet_test (teste fixo)."
    )


def main():
    _ensure_src_on_path()
    args = parse_args()
    set_seed(args.seed)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device={device}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "test":
        train_dl, val_dl, test_dl, class_names, num_classes, n_feats, _ = _prepare_labeled_loaders(args)
        ckpt = args.checkpoint or str(out / "finetuned_best.pt")
        model = build_model(args.model, num_classes, n_feats, args.hidden_lstm).to(device)
        trainer = Trainer(model, device, out)
        res = trainer.evaluate(test_dl, ckpt)
        metrics = classification_metrics_numpy(res["y_true"], res["y_pred"])
        print(f"checkpoint={ckpt}")
        print(f"test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f}")
        return

    train_dl, val_dl, test_dl, class_names, num_classes, n_feats, _ = _prepare_labeled_loaders(args)
    print(f"classes={len(class_names)} n_tsfel_feats={n_feats}")

    model = build_model(args.model, num_classes, n_feats, args.hidden_lstm).to(device)
    trainer = Trainer(model, device, out)

    if args.mode == "supervised":
        lr = args.lr if args.lr is not None else 1e-3
        trainer.train_supervised(
            train_dl,
            val_dl,
            num_classes,
            epochs=args.epochs,
            lr=lr,
            use_class_weights=not args.no_class_weights,
        )
        res = trainer.evaluate(test_dl, out / "supervised_best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        print(f"test: acc={m['accuracy']:.4f} macro_f1={m['f1_macro']:.4f}")
        return

    if args.mode == "finetune":
        load_from = args.checkpoint or (out / "supervised_best.pt")
        lr = args.lr if args.lr is not None else 1e-4
        if trainer.finetune(
            train_dl, val_dl, load_path=load_from, epochs=args.epochs, lr=lr
        ) is None:
            raise SystemExit(f"Fine-tune cancelado: não achou checkpoint em {load_from}")
        res = trainer.evaluate(test_dl, out / "finetuned_best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        print(f"test: acc={m['accuracy']:.4f} macro_f1={m['f1_macro']:.4f}")
        return

    if args.mode == "fixmatch":
        if not args.unlabeled_parquet:
            raise SystemExit("fixmatch exige --unlabeled_parquet")
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
        print(f"test: acc={m['accuracy']:.4f} macro_f1={m['f1_macro']:.4f}")
        return


if __name__ == "__main__":
    main()
