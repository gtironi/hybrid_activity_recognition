"""Linear probe: trains logistic regression over the frozen encoder embeddings.

Goal: quantify what the encoder has learned, in isolation from the TSFEL branch
and from any non-linear fusion head. If the linear probe's macro F1 is much
lower than the TSFEL-only baseline, the encoder is not learning useful
representations (no matter how the hybrid model scores).

This script is intentionally kept outside of ``src/`` — it depends on
``hybrid_activity_recognition`` only to rebuild the model architecture and
load checkpoints; it is not part of the training pipeline.

Usage:
    PYTHONPATH=src python probes/linear_probe.py \
        --checkpoint experiments/0.2_val/cnn_lstm_hybrid_.../best.pt \
        --model cnn_lstm \
        --input_mode hybrid \
        --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
        --labeled_parquet_test  dataset/processed/AcTBeCalf/windowed_test.parquet \
        --output probes/results/cnn_lstm_hybrid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hybrid_activity_recognition.data.dataloader import prepare_train_val_test_loaders  # noqa: E402
from hybrid_activity_recognition.models import build_hybrid_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, required=True,
                   help="Encoder name used in build_hybrid_model (e.g. cnn_lstm, patchtst, robust).")
    p.add_argument("--input_mode", type=str, default="hybrid",
                   help="Match the input_mode the checkpoint was trained with.")
    p.add_argument("--head", type=str, default=None, help="Head name; only needed if non-default.")
    p.add_argument("--labeled_parquet_train", type=str, required=True)
    p.add_argument("--labeled_parquet_test", type=str, required=True)
    p.add_argument("--labeled_parquet_val", type=str, default="")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, required=True,
                   help="Path to write JSON metrics.")
    # PatchTST geometry — used only when building a PatchTST model.
    p.add_argument("--context_length", type=int, default=75)
    p.add_argument("--patch_len", type=int, default=12)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--revin", type=int, default=1, choices=(0, 1))
    p.add_argument("--hidden_lstm", type=int, default=None)
    return p.parse_args()


def encoder_kwargs_from_args(args) -> dict:
    kw: dict = {}
    if args.model == "patchtst":
        kw.update(
            context_length=args.context_length,
            patch_len=args.patch_len,
            stride=args.stride,
            num_layers=args.n_layers,
            num_heads=args.n_heads,
            d_model=args.d_model,
            ffn_dim=args.d_ff,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            revin=bool(args.revin),
        )
    if args.hidden_lstm is not None:
        kw["hidden_lstm"] = args.hidden_lstm
    return kw


@torch.no_grad()
def extract_embeddings(model, dl, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    encoder = model.encoder
    embs, labels = [], []
    for x_sig, _x_feat, y in dl:
        x_sig = x_sig.to(device)
        if hasattr(encoder, "forward_hidden"):
            z = encoder.forward_hidden(x_sig)
        else:
            z = encoder(x_sig)
        if z.ndim > 2:
            z = z.flatten(start_dim=1)
        embs.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl, class_names, num_classes, n_feats, _ = prepare_train_val_test_loaders(
        args.labeled_parquet_train,
        args.labeled_parquet_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
        val_fraction=args.val_fraction,
        parquet_val_path=(args.labeled_parquet_val or None),
    )

    model = build_hybrid_model(
        encoder_name=args.model,
        input_mode=args.input_mode,
        num_classes=num_classes,
        n_tsfel_feats=n_feats,
        head_name=args.head,
        **encoder_kwargs_from_args(args),
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

    print("[info] extracting train embeddings...")
    Z_tr, y_tr = extract_embeddings(model, train_dl, device)
    print("[info] extracting test embeddings...")
    Z_te, y_te = extract_embeddings(model, test_dl, device)
    print(f"[info] train_emb_shape={Z_tr.shape} test_emb_shape={Z_te.shape}")

    scaler = StandardScaler().fit(Z_tr)
    Z_tr_n = scaler.transform(Z_tr)
    Z_te_n = scaler.transform(Z_te)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(Z_tr_n, y_tr)
    y_pred = clf.predict(Z_te_n)

    metrics = {
        "checkpoint": str(ckpt_path),
        "model": args.model,
        "input_mode": args.input_mode,
        "embedding_dim": int(Z_tr.shape[1]),
        "n_train": int(Z_tr.shape[0]),
        "n_test": int(Z_te.shape[0]),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
    }
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
