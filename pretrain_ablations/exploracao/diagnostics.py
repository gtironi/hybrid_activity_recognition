"""Diagnostic plots and checks for each dataset.

Usage:
    python diagnostics.py --dataset actbecalf_windowed --all
    python diagnostics.py --dataset dog_w50_w10 --balance --leakage
    python diagnostics.py --dataset har_tfc --balance --nans
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
# Loaders
# ---------------------------------------------------------------------------

def _get_labels_groups(dataset_id: str, registry: dict) -> tuple[list, list]:
    """Returns (labels, groups) for the full dataset (train + test if available)."""
    info = registry[dataset_id]
    task = info.get("task", "classification")
    if task != "classification":
        return [], []

    if dataset_id == "har_tfc":
        import torch
        root = repo_root() / info["root"]
        labels, groups = [], []
        for split in ["train", "val", "test"]:
            pt = root / f"{split}.pt"
            if not pt.exists():
                continue
            d = torch.load(pt, weights_only=False)
            labels.extend(d["labels"].numpy().tolist())
            # subject IDs not stored in .pt; mark as -1
            groups.extend([-1] * len(d["labels"]))
        id2label = {i: n for i, n in enumerate(info["label_names"])}
        labels = [id2label.get(l, str(l)) for l in labels]
        return labels, groups

    elif dataset_id == "actbecalf_windowed":
        import pandas as pd
        root = repo_root() / info["root"]
        frames = []
        for f in ["train.parquet", "test.parquet"]:
            p = root / f
            if p.exists():
                frames.append(pd.read_parquet(p)[["label", "calf_id"]])
        if not frames:
            return [], []
        df = pd.concat(frames, ignore_index=True)
        return df["label"].tolist(), df["calf_id"].tolist()

    elif dataset_id.startswith("dog"):
        import pandas as pd
        source_csv = info.get("source_csv") or info.get("default_source", "dog_w50.csv")
        root = repo_root() / info["root"]
        df = pd.read_csv(root / source_csv, usecols=["DogID", "label"] if "label" in
                         list(pd.read_csv(root / source_csv, nrows=0).columns) else ["DogID"])
        if "label" not in df.columns:
            return [], df["DogID"].tolist()
        return df["label"].tolist(), df["DogID"].tolist()

    return [], []


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_balance(dataset_id: str, labels: list, out_dir: Path, ts: str) -> None:
    if not labels:
        print("  [balance] No labels available.")
        return
    from collections import Counter
    counts = Counter(labels)
    classes = sorted(counts.keys(), key=lambda x: -counts[x])
    values = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.6), 5))
    bars = ax.bar(range(len(classes)), values, color="steelblue")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(f"{dataset_id} | class balance (total={len(labels)})")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    out_path = out_dir / f"{dataset_id}_balance_{ts}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [balance] {len(classes)} classes | min={min(values)} max={max(values)} | saved: {out_path}")


def check_leakage(dataset_id: str, registry: dict, out_dir: Path, ts: str) -> None:
    """Verify no group_id appears in more than one split of the exported .pt files."""
    info = registry[dataset_id]
    processed_root = info.get("processed_root")
    if processed_root is None:
        processed_root = repo_root() / "pretrain_ablations" / "processed" / dataset_id
    else:
        processed_root = repo_root() / processed_root
    processed_root = Path(processed_root)

    if not processed_root.exists():
        print(f"  [leakage] Processed dir not found: {processed_root}. Run export first.")
        return

    import torch
    split_groups: dict[str, set] = {}
    for split in ["train", "val", "test"]:
        pt = processed_root / f"{split}.pt"
        if not pt.exists():
            continue
        d = torch.load(pt, weights_only=False)
        groups = d["groups"].numpy().tolist()
        split_groups[split] = set(g for g in groups if g != -1)

    splits = list(split_groups.keys())
    violations = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_groups[splits[i]] & split_groups[splits[j]]
            if overlap:
                violations.append((splits[i], splits[j], overlap))

    if violations:
        for s1, s2, ov in violations:
            print(f"  [leakage] !! LEAK: {s1} ∩ {s2} = {len(ov)} groups: {sorted(ov)[:5]}...")
    else:
        print(f"  [leakage] OK — no group appears in >1 split "
              f"({', '.join(f'{s}={len(v)}' for s, v in split_groups.items())} groups)")


def check_nans(dataset_id: str, registry: dict) -> None:
    info = registry[dataset_id]

    if dataset_id == "har_tfc":
        import torch
        root = repo_root() / info["root"]
        for split in ["train", "val", "test"]:
            pt = root / f"{split}.pt"
            if not pt.exists():
                continue
            d = torch.load(pt, weights_only=False)
            s = d["samples"]
            nans = s.isnan().sum().item()
            infs = s.isinf().sum().item()
            print(f"  [nans] {split}: NaN={nans} Inf={infs} | shape={tuple(s.shape)}")

    elif dataset_id == "actbecalf_windowed":
        import pandas as pd
        root = repo_root() / info["root"]
        for f in ["train.parquet", "test.parquet"]:
            p = root / f
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            for col in ["acc_x", "acc_y", "acc_z"]:
                if col in df.columns:
                    arr = np.stack(df[col].values)
                    print(f"  [nans] {f} {col}: NaN={np.isnan(arr).sum()} "
                          f"shape={arr.shape} min={arr.min():.3f} max={arr.max():.3f}")

    elif dataset_id.startswith("dog"):
        import pandas as pd
        source_csv = info.get("source_csv") or info.get("default_source", "dog_w50.csv")
        root = repo_root() / info["root"]
        df = pd.read_csv(root / source_csv, nrows=10000)
        for col in info.get("channel_names", []):
            if col in df.columns:
                nans = df[col].isna().sum()
                print(f"  [nans] {col}: NaN={nans} | min={df[col].min():.3f} max={df[col].max():.3f}")


def check_cadence(dataset_id: str, registry: dict, out_dir: Path, ts: str) -> None:
    """Δt histogram — only meaningful if raw CSV has t_sec column."""
    info = registry[dataset_id]
    if not dataset_id.startswith("dog"):
        print("  [cadence] Only implemented for dog datasets.")
        return

    import pandas as pd
    raw_path = repo_root() / info["root"] / "dog_raw.csv"
    if not raw_path.exists():
        print(f"  [cadence] dog_raw.csv not found at {raw_path}")
        return

    df = pd.read_csv(raw_path, usecols=["DogID", "t_sec"], nrows=50000)
    dts = []
    for _, grp in df.groupby("DogID"):
        dts.extend(grp["t_sec"].diff().dropna().tolist())
    dts = np.array(dts)
    dts = dts[(dts > 0) & (dts < 1)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dts, bins=50, color="steelblue")
    ax.set_xlabel("Δt (s)")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset_id} | sampling cadence (Δt)")
    ax.axvline(np.median(dts), color="red", linestyle="--", label=f"median={np.median(dts)*1000:.1f}ms")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{dataset_id}_cadence_{ts}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [cadence] median Δt={np.median(dts)*1000:.1f}ms (~{1/np.median(dts):.0f}Hz) | saved: {out_path}")


def check_corr(dataset_id: str, registry: dict, out_dir: Path, ts: str) -> None:
    """Inter-channel correlation matrix on a sample of windows."""
    info = registry[dataset_id]
    channels = info.get("channel_names", [])
    if len(channels) < 2:
        print("  [corr] Need ≥2 channels.")
        return

    # Build a flat sample per channel
    if dataset_id == "actbecalf_windowed":
        import pandas as pd
        root = repo_root() / info["root"]
        df = pd.read_parquet(root / "train.parquet").head(500)
        data = {col: np.concatenate(df[col].values) for col in ["acc_x", "acc_y", "acc_z"] if col in df.columns}
        mat = np.array([data[c] for c in data])
        ch_names = list(data.keys())
    elif dataset_id.startswith("dog"):
        import pandas as pd
        source_csv = info.get("source_csv") or info.get("default_source", "dog_w50.csv")
        root = repo_root() / info["root"]
        df = pd.read_csv(root / source_csv, nrows=5000)
        present = [c for c in channels if c in df.columns]
        mat = df[present].values.T.astype(np.float32)
        ch_names = present
    else:
        print(f"  [corr] Not implemented for {dataset_id}")
        return

    corr = np.corrcoef(mat)
    fig, ax = plt.subplots(figsize=(max(5, len(ch_names)), max(4, len(ch_names))))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=7)
    for i in range(len(ch_names)):
        for j in range(len(ch_names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{dataset_id} | inter-channel correlation")
    fig.tight_layout()
    out_path = out_dir / f"{dataset_id}_corr_{ts}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [corr] saved: {out_path}")


def check_psd(dataset_id: str, registry: dict, out_dir: Path, ts: str) -> None:
    """Welch PSD for 2-3 windows per class, channel 0."""
    from scipy.signal import welch

    info = registry[dataset_id]
    fs = info.get("sampling_hz") or 50.0

    if dataset_id == "har_tfc":
        import torch
        root = repo_root() / info["root"]
        d = torch.load(root / "train.pt", weights_only=False)
        samples = d["samples"].numpy()[:, 0, :]  # ch0: (N, T)
        labels = d["labels"].numpy()
        id2label = {i: n for i, n in enumerate(info["label_names"])}
        label_strs = [id2label.get(int(l), str(l)) for l in labels]
    elif dataset_id == "actbecalf_windowed":
        import pandas as pd
        root = repo_root() / info["root"]
        df = pd.read_parquet(root / "train.parquet")
        samples = np.stack(df["acc_x"].values)
        label_strs = df["label"].tolist()
    else:
        print(f"  [psd] Not implemented for {dataset_id}")
        return

    unique = sorted(set(label_strs))
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap("tab20", len(unique))
    for i, lbl in enumerate(unique):
        idxs = [j for j, l in enumerate(label_strs) if l == lbl][:3]
        for idx in idxs:
            f, Pxx = welch(samples[idx], fs=fs, nperseg=min(64, samples.shape[1]))
            ax.semilogy(f, Pxx, color=cmap(i), alpha=0.7, label=lbl if idx == idxs[0] else None)
    handles, lbs = ax.get_legend_handles_labels()
    by_label = dict(zip(lbs, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"{dataset_id} | Welch PSD (ch0, 2-3 windows/class)")
    fig.tight_layout()
    out_path = out_dir / f"{dataset_id}_psd_{ts}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [psd] saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset diagnostics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--leakage", action="store_true")
    parser.add_argument("--nans", action="store_true")
    parser.add_argument("--cadence", action="store_true")
    parser.add_argument("--corr", action="store_true")
    parser.add_argument("--psd", action="store_true")
    parser.add_argument("--out_dir", default=str(_HERE / "figures"))
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if args.dataset not in registry:
        sys.exit(f"Unknown dataset '{args.dataset}'")

    ts = make_timestamp()
    out_dir = Path(args.out_dir)

    run_all = args.all
    labels, groups = _get_labels_groups(args.dataset, registry)

    if run_all or args.balance:
        check_balance(args.dataset, labels, out_dir, ts)
    if run_all or args.leakage:
        check_leakage(args.dataset, registry, out_dir, ts)
    if run_all or args.nans:
        check_nans(args.dataset, registry)
    if run_all or args.cadence:
        check_cadence(args.dataset, registry, out_dir, ts)
    if run_all or args.corr:
        check_corr(args.dataset, registry, out_dir, ts)
    if run_all or args.psd:
        check_psd(args.dataset, registry, out_dir, ts)


if __name__ == "__main__":
    main()
