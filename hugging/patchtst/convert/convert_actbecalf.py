#!/usr/bin/env python3
"""AcTBeCalf streaming CSV: windows inside constant-label segments (segId); split by calfId."""

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

COLS = ["accX", "accY", "accZ", "behaviour", "segId", "calfId"]


def _assign_calves(calf_ids: list[int], rng: np.random.Generator, train_f: float, val_f: float):
    """
    Map calfId -> 0 train, 1 val, 2 test. Uses a shuffled list and contiguous slices
    so train|val|test partition every calf exactly once (no empty test when n >= 2).
    """
    uniq = np.array(sorted(set(calf_ids)), dtype=np.int64)
    rng.shuffle(uniq)
    n = len(uniq)
    out: dict[int, int] = {}
    if n == 0:
        return out
    if n == 1:
        return {int(uniq[0]): 0}
    if n == 2:
        # Need at least train + test; val stays empty (training script handles empty val).
        return {int(uniq[0]): 0, int(uniq[1]): 2}

    n_test = max(1, round(n * (1.0 - train_f - val_f)))
    n_val = max(1, round(n * val_f))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        rest = n - n_train
        n_test = max(1, rest // 2)
        n_val = rest - n_test
        if n_val < 1:
            n_val = 1
            n_test = max(1, n - n_train - n_val)

    k = 0
    for _ in range(n_train):
        out[int(uniq[k])] = 0
        k += 1
    for _ in range(n_val):
        out[int(uniq[k])] = 1
        k += 1
    while k < n:
        out[int(uniq[k])] = 2
        k += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="AcTBeCalf.csv",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Read at most this many rows (smoke test)",
    )
    args = parser.parse_args()

    inp = Path(args.input or (repo_root() / "dataset" / "AcTBeCalf.csv")).resolve()
    out_dir = args.out_dir or (package_root() / "standardized" / "actbecalf")

    _cols = COLS
    if args.max_rows:
        df = pd.read_csv(inp, nrows=args.max_rows, usecols=_cols)
    else:
        df = pd.read_csv(inp, usecols=_cols)
    df = df.dropna(subset=["accX", "accY", "accZ", "behaviour", "segId", "calfId"])

    label_names = sorted(df["behaviour"].unique().tolist())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {str(i): label_names[i] for i in range(len(label_names))}

    win = args.window
    stride = args.stride
    c = 3

    # Collect windows first; split only among calves that actually have >=1 window
    # (important with --max_rows: some calves may appear only in short segments).
    windows: list[tuple[int, np.ndarray, int]] = []

    for seg_id, g in df.groupby("segId", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < win:
            continue
        beh = g["behaviour"].iloc[0]
        if (g["behaviour"] != beh).any():
            continue
        calf = int(g["calfId"].iloc[0])
        feats = g[["accX", "accY", "accZ"]].to_numpy(dtype=np.float32)
        y = int(label2id[beh])
        for start in range(0, len(g) - win + 1, stride):
            block = feats[start : start + win]
            windows.append((calf, block, y))

    calves_with_data = sorted({w[0] for w in windows})
    rng = np.random.default_rng(args.split_seed)
    calf_split = _assign_calves(calves_with_data, rng, args.train_frac, args.val_frac)

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    for calf, block, y in windows:
        sid = calf_split[calf]
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
        "dataset_key": "actbecalf",
        "context_length": win,
        "num_channels": c,
        "num_classes": len(label2id),
        "feature_order": "time_major",
        "label2id": label2id,
        "id2label": id2label,
        "split_seed": args.split_seed,
        "group_column": "calfId",
        "notes": "Windows inside segments; one behaviour per segId; split by calfId",
        "source_csv": str(inp),
        "max_rows": args.max_rows,
    }

    save_meta(out_dir, meta)
    write_split_csv(out_dir / "train.csv", y_train, windows_to_feature_matrix(x_train))
    write_split_csv(out_dir / "val.csv", y_val, windows_to_feature_matrix(x_val))
    write_split_csv(out_dir / "test.csv", y_test, windows_to_feature_matrix(x_test))

    print(f"Wrote {out_dir} (train {len(y_train)} val {len(y_val)} test {len(y_test)})")
    print_train_hint(out_dir, preset="actbecalf")


if __name__ == "__main__":
    main()
