#!/usr/bin/env python3
"""
Optional evaluation helpers: t-SNE on hidden states, stubs for pretrain + probe.

Keep heavy or experimental paths out of train_classification_debug.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.manifold import TSNE
from transformers import PatchTSTConfig, PatchTSTForClassification

from hugging.patchtst.io_standard import load_csv_tensors, load_meta, package_root


def _pool_representation(
    hidden_last: torch.Tensor,
    *,
    pooling: Literal["mean", "cls"],
) -> np.ndarray:
    """
    HF PatchTST hidden_states[-1] can be (B, L, D) or, with channel-independence,
    (B, n_channels, n_patches, D). Pool to (B, D).
    """
    if hidden_last.dim() == 4:
        if pooling == "cls":
            return hidden_last[:, :, 0, :].mean(dim=1).cpu().numpy()
        return hidden_last.mean(dim=(1, 2)).cpu().numpy()
    if pooling == "mean":
        return hidden_last.mean(dim=1).cpu().numpy()
    return hidden_last[:, 0, :].cpu().numpy()


def tsne_from_classification_checkpoint(
    *,
    model_pt: Path,
    data_dir: Path,
    model: PatchTSTForClassification | None = None,
    split: str = "test",
    max_samples: int = 2000,
    perplexity: float = 30.0,
    seed: int = 42,
    pooling: Literal["mean", "cls"] = "mean",
    output_png: Path | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    write_sidecar_json: bool = True,
    plot_title: str | None = None,
) -> Path:
    """
    Load PatchTSTForClassification weights (unless ``model`` is already provided),
    run forward with output_hidden_states=True, pool tokens, fit t-SNE, save PNG.

    Modes:
    - **classification_finetuned**: this function — full model after supervised training.
    - **encoder_only**: not implemented here; use PatchTSTModel + saved encoder weights
      or strip the classification head in a separate script.
    """
    data_dir = Path(data_dir)
    model_pt = Path(model_pt)
    meta = load_meta(data_dir)
    t_ctx = int(meta["context_length"])
    n_ch = int(meta["num_channels"])

    if model is None:
        ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
        config = PatchTSTConfig(**ckpt["config"])
        model = PatchTSTForClassification(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
    model.eval()

    x, y = load_csv_tensors(data_dir / f"{split}.csv", meta)
    if len(y) == 0:
        raise RuntimeError(f"{split} split is empty")
    if len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        x, y = x[idx], y[idx]

    feats = []
    labels = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(x[i : i + bs]).float().to(device)
            yb = y[i : i + bs]
            past = xb.view(xb.size(0), t_ctx, n_ch)
            out = model(past_values=past, output_hidden_states=True)
            hs = out.hidden_states[-1]
            rep = _pool_representation(hs, pooling=pooling)
            feats.append(rep)
            labels.append(yb)
    X = np.concatenate(feats, axis=0)
    y_all = np.concatenate(labels, axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(X) // 4)),
        random_state=seed,
        init="random",
        learning_rate="auto",
    )
    z = tsne.fit_transform(X)

    out_path = Path(
        output_png or (Path(model_pt).parent / f"tsne_{split}_{pooling}.png")
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=y_all, cmap="tab20", s=8, alpha=0.7)
        plt.colorbar(scatter, label="label id")
        plt.title(
            plot_title
            or f"t-SNE ({pooling} pool) — {data_dir.name} / {split}"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except ImportError as e:
        raise RuntimeError("matplotlib required for t-SNE plot") from e

    if write_sidecar_json:
        sidecar = out_path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "model_pt": str(model_pt),
                    "data_dir": str(data_dir),
                    "split": split,
                    "max_samples": int(len(y_all)),
                    "pooling": pooling,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return out_path


def pretrain_then_linear_probe_stub() -> None:
    """
    Planned: PatchTSTForPretraining on unlabeled windows, then train a linear head or
    fine-tune PatchTSTForClassification.

    Not implemented in the debug pipeline; use Hugging Face
    ``PatchTSTForPretraining`` + your saved encoder weights, or extend this module.
    """
    raise NotImplementedError(
        "pretrain_then_linear_probe_stub: add PatchTSTForPretraining loop and checkpoint "
        "loading before supervised head training."
    )


def load_tfc_pt_windows_stub(pt_path: Path) -> None:
    """
    Optional future: load ``train.pt`` / ``test.pt`` from TFC-style preprocessing
    (e.g. ``data/processed/HAR``) and emit the standardized CSV layout via io_standard.

    Not implemented; TFC tensors are often (N, C, T) — transpose to (N, T, C) before flatten.
    """
    raise NotImplementedError(
        f"TFC tensor export not implemented; if you add {pt_path}, convert (N,C,T) -> (N,T,C) "
        "then call write_split_csv."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tsne = sub.add_parser("tsne", help="t-SNE from trained classification checkpoint")
    p_tsne.add_argument("--model_pt", type=Path, required=True)
    p_tsne.add_argument("--data_dir", type=Path, default=None)
    p_tsne.add_argument("--preset", type=str, default=None)
    p_tsne.add_argument("--split", type=str, default="test")
    p_tsne.add_argument("--max_samples", type=int, default=2000)
    p_tsne.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    p_tsne.add_argument("--output_png", type=Path, default=None)
    p_tsne.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.cmd == "tsne":
        data_dir = args.data_dir
        if data_dir is None:
            if args.preset is None:
                raise SystemExit("Provide --data_dir or --preset")
            preset_dirs = {
                "dog_w10": "standardized/dog_w10",
                "dog_w50": "standardized/dog_w50",
                "dog_w100": "standardized/dog_w100",
                "dog_raw": "standardized/dog_raw",
                "actbecalf": "standardized/actbecalf",
                "har": "standardized/har_uci",
                "ettm1": "standardized/ettm1_hour",
            }
            rel = preset_dirs.get(args.preset)
            if not rel:
                raise SystemExit(f"Unknown preset {args.preset!r}")
            data_dir = package_root() / rel
        path = tsne_from_classification_checkpoint(
            model_pt=args.model_pt,
            data_dir=Path(data_dir),
            split=args.split,
            max_samples=args.max_samples,
            pooling=args.pooling,
            output_png=args.output_png,
            device=args.device,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
