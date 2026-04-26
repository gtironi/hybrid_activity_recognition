#!/usr/bin/env python3
"""UCI HAR: build (128, 9) windows from Inertial Signals (not X_train 561-d features)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hugging.patchtst.io_standard import (
    package_root,
    print_train_hint,
    repo_root,
    save_meta,
    windows_to_feature_matrix,
    write_split_csv,
)

# Channel order matches TFC / common UCI layout: body acc(3), body gyro(3), total acc(3)
INERTIAL_FILES = [
    "body_acc_x_{split}.txt",
    "body_acc_y_{split}.txt",
    "body_acc_z_{split}.txt",
    "body_gyro_x_{split}.txt",
    "body_gyro_y_{split}.txt",
    "body_gyro_z_{split}.txt",
    "total_acc_x_{split}.txt",
    "total_acc_y_{split}.txt",
    "total_acc_z_{split}.txt",
]


def _load_inertial_matrix(har_dir: Path, split: str) -> np.ndarray:
    """Return (N, 128, 9)."""
    base = har_dir / split / "Inertial Signals"
    channels = []
    for tmpl in INERTIAL_FILES:
        path = base / tmpl.format(split=split)
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append([float(x) for x in line.split()])
        channels.append(np.array(rows, dtype=np.float32))
    # (9, N, 128) -> (N, 128, 9)
    stacked = np.stack(channels, axis=0)
    return np.transpose(stacked, (1, 2, 0))


def _load_labels(har_dir: Path, split: str) -> np.ndarray:
    path = har_dir / split / f"y_{split}.txt"
    y = np.loadtxt(path, dtype=np.int64)
    if y.ndim == 0:
        y = np.array([int(y)], dtype=np.int64)
    # UCI uses 1..6 -> 0..5
    return y.flatten() - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--har_dir",
        type=Path,
        default=None,
        help="Path to 'HAR UCI' folder (contains train/ test/)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output standardized directory",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of official train used for validation",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    har_dir = args.har_dir or (repo_root() / "data" / "HAR UCI")
    out_dir = args.out_dir or (package_root() / "standardized" / "har_uci")

    x_tr = _load_inertial_matrix(har_dir, "train")
    y_tr = _load_labels(har_dir, "train")
    x_te = _load_inertial_matrix(har_dir, "test")
    y_te = _load_labels(har_dir, "test")

    assert x_tr.shape[0] == len(y_tr), (x_tr.shape, y_tr.shape)
    assert x_te.shape[0] == len(y_te), (x_te.shape, y_te.shape)
    t_len, c = x_tr.shape[1], x_tr.shape[2]

    rng = np.random.default_rng(args.split_seed)
    n_tr = len(y_tr)
    idx = np.arange(n_tr)
    rng.shuffle(idx)
    n_val = max(1, int(n_tr * args.val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    x_val, y_val = x_tr[val_idx], y_tr[val_idx]
    x_train, y_train = x_tr[train_idx], y_tr[train_idx]

    names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    label2id = {n: i for i, n in enumerate(names)}
    id2label = {str(i): names[i] for i in range(6)}

    meta = {
        "dataset_key": "har_uci",
        "context_length": int(t_len),
        "num_channels": int(c),
        "num_classes": 6,
        "feature_order": "time_major",
        "label2id": label2id,
        "id2label": id2label,
        "split_seed": args.split_seed,
        "group_column": None,
        "notes": "UCI HAR Inertial Signals 128x9; labels 0..5; val split from official train",
    }

    save_meta(out_dir, meta)
    write_split_csv(out_dir / "train.csv", y_train, windows_to_feature_matrix(x_train))
    write_split_csv(out_dir / "val.csv", y_val, windows_to_feature_matrix(x_val))
    write_split_csv(out_dir / "test.csv", y_te, windows_to_feature_matrix(x_te))

    print(f"Wrote {out_dir}")
    print_train_hint(out_dir, preset="har")


if __name__ == "__main__":
    main()
