#!/usr/bin/env python3
"""
PatchTST classification (Hugging Face): optional in-run SSL pretrain, then supervised fine-tune.

- Default: random init → supervised only.
- ``--pretrain``: train ``PatchTSTForPretraining`` (masked) on unlabeled windows, save ``pretrain.pt``,
  load encoder into the classifier, then supervised.
- ``--pretrain_ckpt PATH``: skip SSL in this run; load encoder from checkpoint, then supervised.
  If both ``--pretrain`` and ``--pretrain_ckpt`` are set, the checkpoint wins (SSL not re-trained).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig, PatchTSTForClassification, PatchTSTForPretraining

from hugging.patchtst.io_standard import load_csv_tensors, load_meta, package_root

PRESET_DIRS = {
    "dog_w10": "standardized/dog_w10",
    "dog_w50": "standardized/dog_w50",
    "dog_w100": "standardized/dog_w100",
    "dog_raw": "standardized/dog_raw",
    "actbecalf": "standardized/actbecalf",
    "har": "standardized/har_uci",
    "ettm1": "standardized/ettm1_hour",
}


def _resolve_data_dir(preset: str | None, data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).resolve()
    if not preset:
        raise ValueError("Provide --data_dir or --preset")
    rel = PRESET_DIRS.get(preset)
    if not rel:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESET_DIRS)}")
    return (package_root() / rel).resolve()


def _batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def _patchtst_config_for_run(
    args: argparse.Namespace,
    *,
    n_ch: int,
    t_ctx: int,
    num_targets: int,
) -> PatchTSTConfig:
    """
    Single backbone + head layout for both ``PatchTSTForPretraining`` and
    ``PatchTSTForClassification`` (same pattern as one ``PatchTSTConfig`` in HF examples).
    """
    return PatchTSTConfig(
        num_input_channels=n_ch,
        num_targets=num_targets,
        context_length=t_ctx,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        ffn_dim=args.ffn_dim,
        use_cls_token=True,
        head_dropout=0.3,
        do_mask_input=True,
        mask_type="random",
        random_mask_ratio=args.mask_ratio,
    )


def _run_ssl_pretrain_and_save(
    *,
    args: argparse.Namespace,
    config: PatchTSTConfig,
    x_tr: np.ndarray,
    x_va: np.ndarray,
    out: Path,
    to_past,
) -> Path:
    """Train ``PatchTSTForPretraining(config)``; write ``out/pretrain.pt``. Returns path."""
    blocks: list[np.ndarray] = [x_tr]
    if args.use_val_unlabeled and len(x_va) > 0:
        blocks.append(x_va)
    x_all = np.concatenate(blocks, axis=0)
    if len(x_all) == 0:
        raise RuntimeError("no windows for SSL pretrain")

    pt_model = PatchTSTForPretraining(config).to(args.device)
    ds = TensorDataset(torch.from_numpy(x_all).float())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    opt_pt = torch.optim.AdamW(pt_model.parameters(), lr=args.pretrain_lr)

    global_step = 0
    for epoch in range(args.pretrain_epochs):
        pt_model.train()
        losses: list[float] = []
        for (xb,) in loader:
            xb = xb.to(args.device)
            past = to_past(xb)
            opt_pt.zero_grad(set_to_none=True)
            outputs = pt_model(past_values=past)
            loss = outputs.loss
            loss.backward()
            opt_pt.step()
            losses.append(loss.item())
            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"[pretrain] epoch {epoch+1}/{args.pretrain_epochs} step {global_step} "
                    f"loss {np.mean(losses[-args.log_every:]):.4f}"
                )
        print(f"[pretrain] epoch {epoch+1} loss {np.mean(losses):.4f}")

    ckpt_path = out / "pretrain.pt"
    torch.save(
        {"model_state_dict": pt_model.state_dict(), "config": config.to_dict()},
        ckpt_path,
    )
    del pt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=str, default=None, choices=sorted(PRESET_DIRS.keys()))
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch_length", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--no_tsne",
        action="store_true",
        help="Skip all t-SNE PNGs (tsne_before.png and tsne.png)",
    )
    parser.add_argument(
        "--no_tsne_before",
        action="store_true",
        help="Skip tsne_before.png only (random init); still write tsne.png after training unless --no_tsne",
    )
    parser.add_argument(
        "--tsne_max_samples",
        type=int,
        default=2000,
        help="Max test points for t-SNE (subsampled if larger)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Run masked SSL pretrain in this process, then load encoder and fine-tune supervised",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=10,
        help="SSL epochs when --pretrain (ignored if --pretrain_ckpt is set)",
    )
    parser.add_argument(
        "--pretrain_lr",
        type=float,
        default=1e-3,
        help="Learning rate for SSL phase",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="Random mask ratio for PatchTSTForPretraining",
    )
    parser.add_argument(
        "--use_val_unlabeled",
        action="store_true",
        help="Include val split features (no labels) in SSL pretrain data",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=Path,
        default=None,
        help="Load encoder from this pretrain.pt only (no SSL training in this run). Overrides --pretrain.",
    )
    parser.add_argument(
        "--skip_pretrain_config_check",
        action="store_true",
        help="Do not verify pretrain vs classifier PatchTSTConfig encoder fields",
    )
    args = parser.parse_args()

    if args.pretrain and args.pretrain_ckpt:
        print("Note: --pretrain_ckpt set; skipping in-run SSL (loading checkpoint only).")

    data_dir = _resolve_data_dir(args.preset, args.data_dir)
    meta = load_meta(data_dir)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    preset_tag = args.preset or data_dir.name
    out = args.output_dir or (
        package_root() / "runs" / f"{ts}_{preset_tag}_seed{args.seed}"
    )
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_tr, y_tr = load_csv_tensors(data_dir / "train.csv", meta)
    x_va, y_va = load_csv_tensors(data_dir / "val.csv", meta)
    x_te, y_te = load_csv_tensors(data_dir / "test.csv", meta)

    if len(y_tr) == 0:
        raise RuntimeError("train split is empty; re-run converter with different params")

    num_targets = int(meta.get("num_classes", len(meta["label2id"])))
    t_ctx = int(meta["context_length"])
    n_ch = int(meta["num_channels"])

    if x_tr.shape[1] != t_ctx * n_ch:
        raise ValueError(
            f"CSV width {x_tr.shape[1]} != context_length * num_channels = {t_ctx * n_ch}"
        )

    def to_past(x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        return x.view(b, t_ctx, n_ch)

    # ===============================
    # PatchTST for Classification
    # ===============================
    # One config for SSL backbone and classifier (HF-style); manual loops, no Trainer.
    config = _patchtst_config_for_run(
        args, n_ch=n_ch, t_ctx=t_ctx, num_targets=num_targets
    )

    ssl_ckpt_used: Path | None = None
    pretrain_load_source: str | None = None
    if args.pretrain_ckpt:
        ssl_ckpt_used = Path(args.pretrain_ckpt).resolve()
        pretrain_load_source = "external_checkpoint"
    elif args.pretrain:
        ssl_ckpt_used = _run_ssl_pretrain_and_save(
            args=args,
            config=config,
            x_tr=x_tr,
            x_va=x_va,
            out=out,
            to_past=to_past,
        )
        pretrain_load_source = "trained_in_run"

    model = PatchTSTForClassification(config)
    model.to(args.device)

    if ssl_ckpt_used is not None:
        from hugging.patchtst.weights_util import load_pretrain_into_classifier

        n_loaded, skipped = load_pretrain_into_classifier(
            model,
            ssl_ckpt_used,
            check_config=not args.skip_pretrain_config_check,
        )
        payload: dict = {
            "source": pretrain_load_source,
            "checkpoint": str(ssl_ckpt_used),
            "n_tensors_loaded": n_loaded,
            "skipped_keys_sample": skipped[:40],
            "skipped_total": len(skipped),
        }
        if pretrain_load_source == "trained_in_run":
            payload["pretrain_epochs"] = args.pretrain_epochs
        (out / "pretrain_load.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    train_ds = TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(y_tr).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_va),
        torch.from_numpy(y_va).long(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_cfg = {
        "preset": args.preset,
        "data_dir": str(data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "patch_length": args.patch_length,
        "patch_stride": args.patch_stride,
        "d_model": args.d_model,
        "num_hidden_layers": args.num_hidden_layers,
        "meta_context_length": t_ctx,
        "meta_num_channels": n_ch,
        "num_targets": num_targets,
        "pretrain": bool(args.pretrain),
        "pretrain_epochs": args.pretrain_epochs if args.pretrain else None,
        "pretrain_ckpt": str(args.pretrain_ckpt) if args.pretrain_ckpt else None,
        "ssl_weights_from": str(ssl_ckpt_used) if ssl_ckpt_used else None,
    }
    (out / "config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    model_pt = out / "model.pt"
    tsne_import_warned = False

    def try_tsne(
        png: Path,
        m: PatchTSTForClassification,
        plot_title: str,
        err_name: str,
    ) -> None:
        nonlocal tsne_import_warned
        try:
            from hugging.patchtst.extras import tsne_from_classification_checkpoint

            m.eval()
            tsne_from_classification_checkpoint(
                model_pt=model_pt,
                data_dir=data_dir,
                model=m,
                split="test",
                max_samples=args.tsne_max_samples,
                output_png=png,
                device=args.device,
                write_sidecar_json=False,
                plot_title=plot_title,
            )
        except ImportError:
            if not tsne_import_warned:
                print(
                    "matplotlib not installed; skip t-SNE PNGs (pip install matplotlib)"
                )
                tsne_import_warned = True
        except Exception as exc:
            (out / err_name).write_text(str(exc), encoding="utf-8")

    if (
        not args.no_tsne
        and not args.no_tsne_before
        and len(y_te) > 0
    ):
        if ssl_ckpt_used is not None:
            before_title = f"After SSL pretrain, before supervised — {data_dir.name} / test"
        else:
            before_title = f"Before training (random init) — {data_dir.name} / test"
        try_tsne(
            out / "tsne_before.png",
            model,
            plot_title=before_title,
            err_name="tsne_before_error.txt",
        )

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            past = to_past(xb)

            opt.zero_grad(set_to_none=True)
            outputs = model(past_values=past, target_values=yb)
            loss = outputs.loss
            loss.backward()
            opt.step()
            losses.append(loss.item())
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch {epoch+1}/{args.epochs} step {global_step} loss {np.mean(losses[-args.log_every:]):.4f}")

        model.eval()
        with torch.no_grad():
            va_losses, va_acc = [], []
            if len(val_ds) == 0:
                va_losses.append(float("nan"))
                va_acc.append(float("nan"))
            else:
                for xb, yb in val_loader:
                    past = to_past(xb.to(args.device))
                    yb = yb.to(args.device)
                    outputs = model(past_values=past, target_values=yb)
                    va_losses.append(outputs.loss.item())
                    va_acc.append(_batch_accuracy(outputs.prediction_logits, yb))
        print(
            f"epoch {epoch+1} train_loss {np.mean(losses):.4f} "
            f"val_loss {np.nanmean(va_losses):.4f} val_acc {np.nanmean(va_acc):.4f}"
        )

    # Test + metrics
    model.eval()
    all_pred, all_true = [], []
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te).long()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for xb, yb in test_loader:
            past = to_past(xb.to(args.device))
            outputs = model(past_values=past)
            pred = outputs.prediction_logits.argmax(dim=-1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())
    if not all_pred:
        y_pred = np.array([], dtype=np.int64)
        y_true = np.array([], dtype=np.int64)
        acc, f1, cm = float("nan"), float("nan"), []
    else:
        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "label_map": meta.get("id2label", meta.get("label2id")),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if cm:
        try:
            import matplotlib.pyplot as plt

            cm_np = np.asarray(cm, dtype=float)
            n_cls = cm_np.shape[0]
            row_sum = cm_np.sum(axis=1, keepdims=True)
            row_sum = np.maximum(row_sum, 1.0)
            cm_row_pct = cm_np / row_sum * 100.0

            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(cm_np, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set_title("Confusion matrix (counts; % of row = true class)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))

            txt_color_thresh = cm_np.max() / 2.0 if cm_np.size and cm_np.max() > 0 else 0
            for i in range(n_cls):
                for j in range(n_cls):
                    cnt = int(cm_np[i, j])
                    pct = cm_row_pct[i, j]
                    ax.text(
                        j,
                        i,
                        f"{cnt}\n({pct:.1f}%)",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if cm_np[i, j] > txt_color_thresh else "black",
                    )

            plt.savefig(out / "confusion_matrix.png", dpi=120, bbox_inches="tight")
            plt.close()
        except ImportError:
            print(
                "matplotlib not installed; skip confusion_matrix.png "
                "(install with: pip install matplotlib)"
            )

    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, model_pt)
    print(f"Saved run to {out}")

    if not args.no_tsne and len(y_te) > 0:
        try_tsne(
            out / "tsne.png",
            model,
            plot_title=f"After training — {data_dir.name} / test",
            err_name="tsne_error.txt",
        )


if __name__ == "__main__":
    main()
