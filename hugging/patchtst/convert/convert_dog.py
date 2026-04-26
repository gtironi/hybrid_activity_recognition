#!/usr/bin/env python3
"""Dog IMU CSV: sliding windows; split by DogID (no subject leakage)."""

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

FEATURE_COLS = [
    "ABack_x",
    "ABack_y",
    "ABack_z",
    "ANeck_x",
    "ANeck_y",
    "ANeck_z",
    "GBack_x",
    "GBack_y",
    "GBack_z",
    "GNeck_x",
    "GNeck_y",
    "GNeck_z",
]


def _assign_groups_fixed(
    ids: np.ndarray, rng: np.random.Generator, train_f: float, val_f: float
) -> dict[int, int]:
    uniq = np.unique(ids)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, round(n * (1.0 - train_f - val_f)))
    n_val = max(1, round(n * val_f))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = max(1, n - n_train - n_test)
    train_set = set(uniq[:n_train].tolist())
    val_set = set(uniq[n_train : n_train + n_val].tolist())
    test_set = set(uniq[n_train + n_val :].tolist())
    assignment = {}
    for u in uniq:
        ui = int(u)
        if u in test_set:
            assignment[ui] = 2
        elif u in val_set:
            assignment[ui] = 1
        else:
            assignment[ui] = 0
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="e.g. data/dog/dog_w10.csv",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    inp = Path(args.input).resolve()
    stem = inp.stem
    out_dir = args.out_dir or (package_root() / "standardized" / stem)

    df = pd.read_csv(inp, usecols=["DogID", "TestNum"] + FEATURE_COLS + ["label"])
    df = df.dropna(subset=FEATURE_COLS + ["label"])

    label_names = sorted(df["label"].unique().tolist())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {str(i): label_names[i] for i in range(len(label_names))}

    rng = np.random.default_rng(args.split_seed)
    dog_ids = df["DogID"].to_numpy()
    split_map = _assign_groups_fixed(dog_ids, rng, args.train_frac, args.val_frac)

    win = args.window
    stride = args.stride
    c = len(FEATURE_COLS)

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    for (dog_id, test_num), g in df.groupby(["DogID", "TestNum"], sort=False):
        g = g.reset_index(drop=True)
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        labs = g["label"].to_numpy()
        sid = split_map[int(dog_id)]
        for start in range(0, len(g) - win + 1, stride):
            wlab = labs[start : start + win]
            if len(np.unique(wlab)) != 1:
                continue
            block = feats[start : start + win]
            y = int(label2id[wlab[0]])
            if sid == 0:
                train_x.append(block)
                train_y.append(y)
            elif sid == 1:
                val_x.append(block)
                val_y.append(y)
            else:
                test_x.append(block)
                test_y.append(y)

    def stack(xs, ys):
        if not xs:
            return np.zeros((0, win, c), dtype=np.float32), np.array([], dtype=np.int64)
        return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)

    x_train, y_train = stack(train_x, train_y)
    x_val, y_val = stack(val_x, val_y)
    x_test, y_test = stack(test_x, test_y)

    meta = {
        "dataset_key": stem,
        "context_length": win,
        "num_channels": c,
        "num_classes": len(label2id),
        "feature_order": "time_major",
        "label2id": label2id,
        "id2label": id2label,
        "split_seed": args.split_seed,
        "group_column": "DogID",
        "notes": f"Sliding windows stride={stride}; uniform label within window; split by DogID",
        "source_csv": str(inp),
    }

    save_meta(out_dir, meta)
    write_split_csv(out_dir / "train.csv", y_train, windows_to_feature_matrix(x_train))
    write_split_csv(out_dir / "val.csv", y_val, windows_to_feature_matrix(x_val))
    write_split_csv(out_dir / "test.csv", y_test, windows_to_feature_matrix(x_test))

    print(f"Wrote {out_dir} (train {len(y_train)} val {len(y_val)} test {len(y_test)})")
    preset = stem if stem in ("dog_w10", "dog_w50", "dog_w100", "dog_raw") else None
    print_train_hint(out_dir, preset=preset)


if __name__ == "__main__":
    main()
