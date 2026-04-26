#!/usr/bin/env python3
"""ETTm1: sliding windows over 7 variates; proxy label hour_of_day (0-23)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hugging.patchtst.io_standard import (
    package_root,
    print_train_hint,
    repo_root,
    save_meta,
    windows_to_feature_matrix,
    write_split_csv,
)

VARIATES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument(
        "--proxy",
        choices=("hour_of_day",),
        default="hour_of_day",
        help="Supervised proxy (no behaviour labels in ETT)",
    )
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    inp = Path(args.input or (repo_root() / "data" / "etth" / "ETTm1.csv")).resolve()
    out_dir = args.out_dir or (package_root() / "standardized" / "ettm1_hour")

    df = pd.read_csv(inp)
    df["date"] = pd.to_datetime(df["date"])
    if args.proxy == "hour_of_day":
        labels_full = df["date"].dt.hour.to_numpy(dtype=np.int64)
    else:
        raise ValueError(args.proxy)

    feats_full = df[VARIATES].to_numpy(dtype=np.float32)
    n, c = feats_full.shape
    win = args.window
    stride = args.stride

    xs = []
    ys = []
    for start in range(0, n - win + 1, stride):
        block = feats_full[start : start + win]
        # label = hour at end of window (clear causal proxy)
        y = int(labels_full[start + win - 1])
        xs.append(block)
        ys.append(y)

    if not xs:
        raise RuntimeError("No windows; check CSV length vs window")

    x_all = np.stack(xs, axis=0)
    y_all = np.array(ys, dtype=np.int64)

    rng = np.random.default_rng(args.split_seed)
    idx = np.arange(len(y_all))
    rng.shuffle(idx)
    n_w = len(idx)
    n_test = max(1, round(n_w * (1.0 - args.train_frac - args.val_frac)))
    n_val = max(1, round(n_w * args.val_frac))
    n_train = n_w - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = max(1, n_w - n_train - n_test)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]

    label2id = {str(h): h for h in range(24)}
    id2label = {str(h): f"hour_{h:02d}" for h in range(24)}

    meta = {
        "dataset_key": "ettm1_hour",
        "context_length": win,
        "num_channels": c,
        "num_classes": 24,
        "feature_order": "time_major",
        "label2id": label2id,
        "id2label": id2label,
        "split_seed": args.split_seed,
        "group_column": None,
        "proxy_task": args.proxy,
        "notes": "ETTm1 has no behaviour labels; hour-of-day at window end as proxy",
        "source_csv": str(inp),
    }

    save_meta(out_dir, meta)
    write_split_csv(out_dir / "train.csv", y_train, windows_to_feature_matrix(x_train))
    write_split_csv(out_dir / "val.csv", y_val, windows_to_feature_matrix(x_val))
    write_split_csv(out_dir / "test.csv", y_test, windows_to_feature_matrix(x_test))

    print(f"Wrote {out_dir} (train {len(y_train)} val {len(y_val)} test {len(y_test)})")
    print_train_hint(out_dir, preset="ettm1")


if __name__ == "__main__":
    main()
