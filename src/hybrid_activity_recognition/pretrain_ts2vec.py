"""TS2Vec pretraining CLI for CNN+LSTM-family encoders.

Usage:
    PYTHONPATH=src python -m hybrid_activity_recognition.pretrain_ts2vec \\
        --model cnn_lstm \\
        --pretrain_parquet dataset/processed/AcTBeCalf/windowed_train.parquet \\
        --output_dir checkpoints/ts2vec/cnn_lstm \\
        --epochs 100

Then in your supervised run:
    --init_encoder_from checkpoints/ts2vec/cnn_lstm/ts2vec_best.pt

PatchTST is not supported here (use its own ``--mode pretrain`` MAE pipeline).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hybrid_activity_recognition.data.pretrain_dataset import prepare_pretrain_dataloader
from hybrid_activity_recognition.models.encoders import CNNLSTMEncoder, RobustCNNLSTMEncoder
from hybrid_activity_recognition.training.ts2vec_pretrain import ts2vec_pretrain_encoder
from hybrid_activity_recognition.utils.logging import setup_logging
from hybrid_activity_recognition.utils.repro import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="TS2Vec pretraining for CNN+LSTM encoders.")
    p.add_argument("--model", choices=("cnn_lstm", "robust"), required=True)
    p.add_argument("--pretrain_parquet", type=str, required=True,
                   help="Windowed parquet (labels are ignored).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temporal_unit", type=int, default=0,
                   help="Minimum scale (in pooling levels) at which to apply temporal contrast.")
    p.add_argument("--hidden_lstm", type=int, default=None,
                   help="Override default LSTM hidden size. MUST match the supervised run.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _build_encoder(name: str, hidden_lstm: int | None):
    cls = {"cnn_lstm": CNNLSTMEncoder, "robust": RobustCNNLSTMEncoder}[name]
    return cls(hidden_lstm=hidden_lstm) if hidden_lstm is not None else cls()


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dl, _, _ = prepare_pretrain_dataloader(
        args.pretrain_parquet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    encoder = _build_encoder(args.model, args.hidden_lstm).to(device)

    ts2vec_pretrain_encoder(
        encoder=encoder,
        dataloader=dl,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temporal_unit=args.temporal_unit,
        output_dir=out,
    )


if __name__ == "__main__":
    main()
