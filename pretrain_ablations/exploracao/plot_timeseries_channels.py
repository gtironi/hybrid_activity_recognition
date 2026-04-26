"""Plot raw or windowed time-series channels for each dataset.

Raw mode:  one figure, one subplot per channel, x = sample index or time.
Windows mode: one figure per window, one subplot per channel.

Usage examples:
    python plot_timeseries_channels.py --dataset har_tfc --mode windows --max_windows 4
    python plot_timeseries_channels.py --dataset actbecalf_windowed --mode raw --group_id 3
    python plot_timeseries_channels.py --dataset dog_w50_w10 --mode raw --group_id 16 --max_rows 500
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
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))

from utils import load_registry, make_timestamp, repo_root


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_har_windows(registry: dict, split: str = "train") -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch
    root = repo_root() / registry["har_tfc"]["root"]
    d = torch.load(root / f"{split}.pt", weights_only=False)
    samples = d["samples"].numpy().astype(np.float32)  # (N, C, T)
    labels = d["labels"].numpy()
    id2label = {i: n for i, n in enumerate(registry["har_tfc"]["label_names"])}
    return samples, labels, [id2label.get(int(l), str(l)) for l in labels]


def _load_actbecalf_windows(registry: dict, group_id: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import pandas as pd
    root = repo_root() / registry["actbecalf_windowed"]["root"]
    df = pd.read_parquet(root / "train.parquet")
    if group_id is not None:
        df = df[df["calf_id"] == group_id].reset_index(drop=True)
    acc_x = np.stack(df["acc_x"].values)
    acc_y = np.stack(df["acc_y"].values)
    acc_z = np.stack(df["acc_z"].values)
    samples = np.stack([acc_x, acc_y, acc_z], axis=1).astype(np.float32)  # (N, 3, T)
    labels = df["label"].values
    return samples, labels, labels.tolist()


def _load_dog_raw(registry: dict, reg_key: str, group_id: int | None = None, max_rows: int = 3000) -> tuple[np.ndarray, list[str], list[str]]:
    import pandas as pd
    info = registry[reg_key]
    csv_name = info.get("source_csv") or info.get("default_source", "dog_w50.csv")
    root = repo_root() / info["root"]
    df = pd.read_csv(root / csv_name, nrows=None)
    if group_id is not None:
        df = df[df["DogID"] == group_id].reset_index(drop=True)
    df = df.head(max_rows)
    channels = info["channel_names"]
    raw = df[channels].values.astype(np.float32).T  # (C, rows)
    labels = df["label"].tolist() if "label" in df.columns else ["?" ] * len(df)
    return raw, channels, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_raw(data: np.ndarray, channel_names: list[str], title: str, out_path: Path) -> None:
    """data: (C, T) — one subplot per channel."""
    C = data.shape[0]
    fig, axes = plt.subplots(C, 1, figsize=(14, 2.5 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        ax.plot(data[i], linewidth=0.6, color=f"C{i}")
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("sample index")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved: {out_path}")


def _plot_window(window: np.ndarray, channel_names: list[str], title: str, out_path: Path, max_channels: int | None = None) -> None:
    """window: (C, T) — one subplot per channel."""
    if max_channels:
        window = window[:max_channels]
        channel_names = channel_names[:max_channels]
    C = window.shape[0]
    fig, axes = plt.subplots(C, 1, figsize=(10, 2.5 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        ax.plot(window[i], linewidth=1.0, color=f"C{i}")
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IMU time-series channels")
    parser.add_argument("--dataset", required=True, help="Key in dataset_registry.yaml")
    parser.add_argument("--mode", choices=["raw", "windows"], default="raw")
    parser.add_argument("--max_rows", type=int, default=3000, help="Max rows in raw mode")
    parser.add_argument("--max_windows", type=int, default=5, help="Max windows to plot in windows mode")
    parser.add_argument("--max_channels", type=int, default=None, help="Limit subplots per figure")
    parser.add_argument("--group_id", type=int, default=None, help="calf_id / DogID to filter on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", default=str(_HERE / "figures"))
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    registry = load_registry(args.registry)
    ts = make_timestamp()
    out_dir = Path(args.out_dir)
    ds = args.dataset

    if ds not in registry:
        sys.exit(f"Unknown dataset '{ds}'. Available: {list(registry.keys())}")

    info = registry[ds]
    task = info.get("task", "classification")

    # -----------------------------------------------------------------------
    # Determine dataset type and load
    # -----------------------------------------------------------------------
    is_har = ds == "har_tfc"
    is_actbecalf = ds == "actbecalf_windowed"
    is_dog = ds.startswith("dog")

    channel_names = info.get("channel_names", [])

    if args.mode == "raw":
        if is_har:
            print("HAR .pt has no continuous time-series. Falling back to windows mode.")
            args.mode = "windows"
        elif is_actbecalf:
            # Parquet: concatenate window arrays along time axis as proxy for "raw"
            samples, labels, label_strs = _load_actbecalf_windows(registry, group_id=args.group_id)
            n = min(len(samples), args.max_rows // samples.shape[2] + 1)
            concat = np.concatenate(samples[:n], axis=1)  # (3, N*T)
            title = f"{ds} | raw (first {n} windows concatenated)"
            if args.max_channels:
                concat = concat[:args.max_channels]
                channel_names = channel_names[:args.max_channels]
            _plot_raw(concat, channel_names, title, out_dir / f"{ds}_channels_raw_{ts}.png")
            return
        elif is_dog:
            raw, ch_names, labels = _load_dog_raw(registry, ds, group_id=args.group_id, max_rows=args.max_rows)
            if args.max_channels:
                raw = raw[:args.max_channels]
                ch_names = ch_names[:args.max_channels]
            title = f"{ds} | raw" + (f" | DogID={args.group_id}" if args.group_id else "")
            _plot_raw(raw, ch_names, title, out_dir / f"{ds}_channels_raw_{ts}.png")
            return
        else:
            sys.exit(f"Raw mode not supported for dataset '{ds}' (task={task})")

    # windows mode
    if is_har:
        samples, labels, label_strs = _load_har_windows(registry, split=args.split)
    elif is_actbecalf:
        samples, labels, label_strs = _load_actbecalf_windows(registry, group_id=args.group_id)
    elif is_dog:
        # For windows mode on dog: build windows from raw CSV using export logic inline
        import pandas as pd
        source_csv = info.get("source_csv") or info.get("default_source", "dog_w50.csv")
        window_len = info.get("window_len", 10)
        root = repo_root() / info["root"]
        df = pd.read_csv(root / source_csv)
        if args.group_id is not None:
            df = df[df["DogID"] == args.group_id].reset_index(drop=True)
        ch = info["channel_names"]
        windows, window_labels = [], []
        for (did, tn), grp in df.groupby(["DogID", "TestNum"]):
            grp = grp.reset_index(drop=True)
            for start in range(0, len(grp) - window_len + 1, max(1, window_len // 2)):
                w = grp.iloc[start:start + window_len]
                lbl = w["label"].mode()[0] if "label" in w.columns else "?"
                windows.append(w[ch].values.T.astype(np.float32))
                window_labels.append(lbl)
        samples = np.array(windows)
        label_strs = window_labels
    else:
        sys.exit(f"Windows mode not supported for dataset '{ds}'")

    n_total = len(samples)
    n_plot = min(args.max_windows, n_total)
    idx = rng.choice(n_total, n_plot, replace=False)

    for i, wi in enumerate(idx):
        window = samples[wi]  # (C, T)
        lbl = label_strs[wi] if wi < len(label_strs) else "?"
        title = f"{ds} | window {wi} | label={lbl}"
        fname = out_dir / f"{ds}_window_{i:03d}_{ts}.png"
        _plot_window(window, list(channel_names), title, fname, max_channels=args.max_channels)

    print(f"Plotted {n_plot} windows.")


if __name__ == "__main__":
    main()
