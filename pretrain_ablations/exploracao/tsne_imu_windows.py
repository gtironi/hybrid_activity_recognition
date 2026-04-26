"""t-SNE visualization of IMU windows using hand-crafted features.

Reads raw parquet/pt sources (--source raw) or exported canonical .pt files
(--source transformed). Uses channel 0 only for feature extraction.

Usage:
    python tsne_imu_windows.py --dataset har_tfc --max_points 3000 --seed 42
    python tsne_imu_windows.py --dataset actbecalf_windowed --max_points 5000
    python tsne_imu_windows.py --dataset har_tfc --source transformed --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from utils import load_registry, make_timestamp, repo_root


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _hand_features(windows: np.ndarray) -> np.ndarray:
    """windows: (N, T) — channel 0 already selected.
    Returns (N, 12) feature matrix.
    """
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-8
    mn = windows.min(axis=1, keepdims=True)
    mx = windows.max(axis=1, keepdims=True)
    rms = np.sqrt((windows ** 2).mean(axis=1, keepdims=True))
    ptp = (mx - mn)
    # top-5 FFT magnitude bins (excluding DC)
    fft_mag = np.abs(np.fft.rfft(windows, axis=1))[:, 1:6]  # (N, 5)
    feats = np.concatenate([mean, std, mn, mx, rms, ptp, fft_mag], axis=1)  # (N, 11)
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Data loaders (channel 0 extracted)
# ---------------------------------------------------------------------------

def _load_raw(dataset_id: str, registry: dict, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    info = registry[dataset_id]
    rng = np.random.default_rng(seed)

    if dataset_id == "har_tfc":
        import torch
        root = repo_root() / info["root"]
        d = torch.load(root / "train.pt", weights_only=False)
        samples = d["samples"].numpy().astype(np.float32)  # (N, C, T)
        labels = d["labels"].numpy().astype(int)
        id2label = {i: n for i, n in enumerate(info["label_names"])}
        label_strs = [id2label.get(l, str(l)) for l in labels]

    elif dataset_id == "actbecalf_windowed":
        import pandas as pd
        root = repo_root() / info["root"]
        df = pd.read_parquet(root / "train.parquet")
        acc_x = np.stack(df["acc_x"].values).astype(np.float32)  # (N, T)
        samples = acc_x[:, np.newaxis, :]  # (N, 1, T)
        label_strs = df["label"].tolist()
        unique = sorted(set(label_strs))
        l2i = {l: i for i, l in enumerate(unique)}
        labels = np.array([l2i[l] for l in label_strs])

    elif dataset_id.startswith("dog"):
        import pandas as pd
        source_csv = info.get("source_csv") or info.get("default_source", "dog_w50.csv")
        window_len = info.get("window_len", 10)
        root = repo_root() / info["root"]
        df = pd.read_csv(root / source_csv)
        ch0 = info["channel_names"][0]
        windows, window_labels = [], []
        for (did, tn), grp in df.groupby(["DogID", "TestNum"]):
            grp = grp.reset_index(drop=True)
            for start in range(0, len(grp) - window_len + 1, window_len):
                w = grp.iloc[start:start + window_len]
                lbl = w["label"].mode()[0] if "label" in w.columns else "?"
                windows.append(w[ch0].values.astype(np.float32))
                window_labels.append(lbl)
        ch0_arr = np.array(windows)  # (N, T)
        samples = ch0_arr[:, np.newaxis, :]
        label_strs = window_labels
        unique = sorted(set(label_strs))
        l2i = {l: i for i, l in enumerate(unique)}
        labels = np.array([l2i[l] for l in label_strs])
    else:
        sys.exit(f"t-SNE not supported for dataset '{dataset_id}'")

    N = len(samples)
    if N > max_points:
        idx = rng.choice(N, max_points, replace=False)
        samples = samples[idx]
        labels = labels[idx]
        label_strs = [label_strs[i] for i in idx]

    return samples[:, 0, :], labels, label_strs  # ch0: (N, T)


def _load_transformed(dataset_id: str, registry: dict, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch
    info = registry[dataset_id]
    processed_root = info.get("processed_root")
    if processed_root is None:
        processed_root = repo_root() / "pretrain_ablations" / "processed" / dataset_id
    else:
        processed_root = repo_root() / processed_root

    pt_path = Path(processed_root) / "train.pt"
    if not pt_path.exists():
        sys.exit(f"Exported .pt not found: {pt_path}\nRun export script first.")

    d = torch.load(pt_path, weights_only=False)
    samples = d["samples"].numpy().astype(np.float32)  # (N, C, T)
    labels = d["labels"].numpy().astype(int)
    meta = d.get("meta", {})
    id2label = meta.get("id2label", {})
    label_strs = [id2label.get(str(l), str(l)) for l in labels]

    rng = np.random.default_rng(seed)
    N = len(samples)
    if N > max_points:
        idx = rng.choice(N, max_points, replace=False)
        samples = samples[idx]
        labels = labels[idx]
        label_strs = [label_strs[i] for i in idx]

    return samples[:, 0, :], labels, label_strs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_tsne(coords: np.ndarray, labels: np.ndarray, label_strs: list[str],
               title: str, out_png: Path, out_csv: Path) -> None:
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_name: dict[int, str] = {}
    for l, s in zip(labels, label_strs):
        label_to_name[int(l)] = s

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ul in enumerate(unique_labels):
        mask = labels == ul
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.5,
                   color=cmap(i), label=label_to_name.get(ul, str(ul)))
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_png}")

    # Save CSV
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label_id", "label_name"])
        for i in range(len(coords)):
            writer.writerow([coords[i, 0], coords[i, 1], int(labels[i]), label_strs[i]])
    print(f"  saved: {out_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE of IMU windows (hand-crafted features)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", choices=["raw", "transformed"], default="raw")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=str(_HERE / "figures"))
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if args.dataset not in registry:
        sys.exit(f"Unknown dataset '{args.dataset}'")

    ts = make_timestamp()
    out_dir = Path(args.out_dir)
    source_tag = args.source

    print(f"Loading {args.dataset} ({source_tag})...")
    if args.source == "raw":
        ch0, labels, label_strs = _load_raw(args.dataset, registry, args.max_points, args.seed)
    else:
        ch0, labels, label_strs = _load_transformed(args.dataset, registry, args.max_points, args.seed)

    print(f"  {len(ch0)} windows, T={ch0.shape[1]}, {len(set(labels))} classes")

    # Feature extraction
    feats = _hand_features(ch0)  # (N, 11)

    # PCA pre-reduction if T is large
    if feats.shape[1] > 50:
        from sklearn.decomposition import PCA
        feats = PCA(n_components=50, random_state=args.seed).fit_transform(feats)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    feats = StandardScaler().fit_transform(feats)

    # t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity}, n={len(feats)})...")
    from sklearn.manifold import TSNE
    coords = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, len(feats) - 1),
        random_state=args.seed,
        n_iter=1000,
    ).fit_transform(feats)

    title = f"{args.dataset} | t-SNE ({source_tag}) | n={len(feats)}"
    out_png = out_dir / f"{args.dataset}_tsne_{source_tag}_{ts}.png"
    out_csv = out_dir / f"{args.dataset}_tsne_{source_tag}_{ts}.csv"
    _plot_tsne(coords, labels, label_strs, title, out_png, out_csv)


if __name__ == "__main__":
    main()
