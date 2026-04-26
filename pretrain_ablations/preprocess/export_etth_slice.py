"""Export ETTm1.csv to canonical .pt format for PatchTST/PatchTSMixer MAE sanity check.

No labels. Chronological 70/10/20 train/val/test split. OT channel only.
task=forecasting_sanity — NOT for classification pipeline.

Usage:
    python export_etth_slice.py --window_len 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from utils import hash_tensor, load_registry, make_timestamp, repo_root, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    registry = load_registry()
    root = repo_root()
    csv_path = root / "data" / "etth" / "ETTm1.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    out_root = root / "pretrain_ablations" / "processed" / "etth"
    out_root.mkdir(parents=True, exist_ok=True)
    ts = make_timestamp()

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, columns: {df.columns.tolist()}")

    values = df["OT"].values.astype(np.float32)  # (T_total,)
    N_total = len(values)

    # Build windows (stride=1 for forecasting sanity, stride can be increased)
    windows = []
    for start in range(0, N_total - args.window_len + 1, args.stride):
        windows.append(values[start:start + args.window_len])
    windows = np.array(windows, dtype=np.float32)[:, np.newaxis, :]  # (N, 1, T)
    N = len(windows)
    print(f"  {N} windows of length {args.window_len}")

    # Chronological split 70/10/20
    n_train = int(N * 0.70)
    n_val = int(N * 0.10)
    n_test = N - n_train - n_val

    splits = {
        "train": windows[:n_train],
        "val": windows[n_train:n_train + n_val],
        "test": windows[n_train + n_val:],
    }

    for split_name, X in splits.items():
        t_samples = torch.tensor(X, dtype=torch.float32)
        t_labels = torch.full((len(X),), -1, dtype=torch.long)
        t_groups = torch.full((len(X),), -1, dtype=torch.long)

        meta = {
            "label2id": {},
            "id2label": {},
            "channel_names": ["OT"],
            "channel0_physical": "OT",
            "num_channels": 1,
            "dataset_id": "etth",
            "sampling_hz": None,
            "window_len": args.window_len,
            "split": split_name,
            "split_policy": "chronological",
            "task": "forecasting_sanity",
            "export_timestamp": ts,
            "sha256_samples": hash_tensor(t_samples),
        }
        canonical = {"samples": t_samples, "labels": t_labels, "groups": t_groups, "meta": meta}
        torch.save(canonical, out_root / f"{split_name}.pt")
        print(f"  [{split_name}] shape={tuple(t_samples.shape)}")

    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
