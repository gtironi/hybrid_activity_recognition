"""Export dog CSV data to canonical .pt format with sliding window.

Each row in dog_w*.csv is ONE raw timestep. Apply sliding window per (DogID, TestNum) group.
Only windows where all rows share the same label are kept (100% purity).

Usage:
    python export_dog.py --source_csv dog_w50.csv --window_len 10 --stride 5
    python export_dog.py --source_csv dog_w10.csv --window_len 20 --stride 10
    python export_dog.py --source_csv dog_w100.csv --window_len 10 --stride 5
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from utils import hash_tensor, load_registry, make_timestamp, repo_root, save_json

ACC_CHANNELS = ["ABack_x", "ABack_y", "ABack_z", "ANeck_x", "ANeck_y", "ANeck_z"]
GYRO_CHANNELS = ["GBack_x", "GBack_y", "GBack_z", "GNeck_x", "GNeck_y", "GNeck_z"]


def sliding_windows(df_group: pd.DataFrame, channels: list[str], window_len: int, stride: int
                    ) -> tuple[list[np.ndarray], list[str]]:
    """Apply sliding window over one (DogID, TestNum) group.
    Returns list of (C, window_len) arrays and corresponding labels.
    Only keeps windows where all rows share the same label.
    """
    data = df_group[channels].values.astype(np.float32)  # (T_group, C)
    labels = df_group["label"].values if "label" in df_group.columns else None
    N = len(data)
    windows, window_labels = [], []
    for start in range(0, N - window_len + 1, stride):
        end = start + window_len
        w = data[start:end]  # (window_len, C)
        if labels is not None:
            lbls = labels[start:end]
            unique_lbls = np.unique(lbls)
            if len(unique_lbls) != 1:
                continue  # mixed label — skip
            window_labels.append(str(unique_lbls[0]))
        else:
            window_labels.append("unknown")
        windows.append(w.T)  # (C, window_len)
    return windows, window_labels


def _write_split(samples: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 split: str, out_root: Path, meta_base: dict, ts: str) -> None:
    N, C, T = samples.shape
    t_samples = torch.tensor(samples, dtype=torch.float32)
    t_labels = torch.tensor(labels, dtype=torch.long)
    t_groups = torch.tensor(groups, dtype=torch.long)

    cnt = Counter(labels.tolist())
    id2label: dict = meta_base["id2label"]
    class_dist = {id2label.get(str(k), str(k)): v for k, v in sorted(cnt.items())}

    meta = {**meta_base, "split": split, "export_timestamp": ts,
            "sha256_samples": hash_tensor(t_samples)}
    canonical = {"samples": t_samples, "labels": t_labels, "groups": t_groups, "meta": meta}

    out_root.mkdir(parents=True, exist_ok=True)
    torch.save(canonical, out_root / f"{split}.pt")
    print(f"  [{split}] shape={tuple(t_samples.shape)} labels={len(t_labels)} "
          f"dogs={len(set(groups.tolist()))}")

    splits_dir = out_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    save_json({
        "indices": list(range(N)),
        "class_distribution": class_dist,
        "group_ids": sorted(set(groups.tolist())),
        "n_samples": N,
    }, splits_dir / f"{split}_indices.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="dog_w50.csv")
    parser.add_argument("--window_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--include_gyro", action="store_true")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    registry = load_registry()
    root = repo_root()
    dog_root = root / "data" / "dog"
    csv_path = dog_root / args.source_csv
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    channels = ACC_CHANNELS + (GYRO_CHANNELS if args.include_gyro else [])
    source_tag = args.source_csv.replace(".csv", "").lstrip("dog_")
    out_name = f"dog_{source_tag}_w{args.window_len}"
    out_root = root / "pretrain_ablations" / "processed" / out_name
    ts = make_timestamp()

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, columns: {df.columns.tolist()}")

    # Only keep columns we need
    keep_cols = ["DogID", "TestNum"] + [c for c in channels if c in df.columns]
    if "label" in df.columns:
        keep_cols.append("label")
    df = df[keep_cols].copy()

    # Sort by DogID, TestNum (row order = time within each group)
    df = df.sort_values(["DogID", "TestNum"]).reset_index(drop=True)

    present_channels = [c for c in channels if c in df.columns]
    if not present_channels:
        sys.exit(f"None of the expected channels found. Available: {df.columns.tolist()}")

    print(f"  Using channels: {present_channels}")
    print(f"  Window len={args.window_len}, stride={args.stride}")

    # Build windows
    all_windows, all_labels, all_dog_ids = [], [], []
    skipped = 0
    for (dog_id, test_num), grp in df.groupby(["DogID", "TestNum"]):
        grp = grp.reset_index(drop=True)
        wins, lbls = sliding_windows(grp, present_channels, args.window_len, args.stride)
        if not wins:
            skipped += 1
            continue
        all_windows.extend(wins)
        all_labels.extend(lbls)
        all_dog_ids.extend([int(dog_id)] * len(wins))

    print(f"  Total windows: {len(all_windows)} (skipped {skipped} groups with no pure windows)")

    if not all_windows:
        sys.exit("No windows produced. Check window_len vs group sizes.")

    X = np.array(all_windows, dtype=np.float32)  # (N, C, T)
    dog_ids = np.array(all_dog_ids, dtype=np.int64)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(all_labels).astype(np.int64)
    label2id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    id2label = {str(int(i)): cls for i, cls in enumerate(le.classes_)}
    save_json(label2id, out_root / "label2id.json")

    print(f"  {len(le.classes_)} classes: {le.classes_.tolist()}")
    cnt = Counter(y.tolist())
    print(f"  Class dist: {sorted(cnt.items(), key=lambda x: -x[1])[:5]} ...")

    # Group split by DogID: first split off test, then val from remaining
    unique_dogs = np.unique(dog_ids)
    n_dogs = len(unique_dogs)
    print(f"  {n_dogs} unique dogs")

    # Test split
    gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups=dog_ids))

    X_test = X[test_idx]; y_test = y[test_idx]; g_test = dog_ids[test_idx]
    X_tv = X[trainval_idx]; y_tv = y[trainval_idx]; g_tv = dog_ids[trainval_idx]

    # Val split from remaining
    gss_val = GroupShuffleSplit(n_splits=1, test_size=args.val_size / (1 - args.test_size), random_state=42)
    train_idx2, val_idx2 = next(gss_val.split(X_tv, y_tv, groups=g_tv))

    X_train = X_tv[train_idx2]; y_train = y_tv[train_idx2]; g_train = g_tv[train_idx2]
    X_val = X_tv[val_idx2]; y_val = y_tv[val_idx2]; g_val = g_tv[val_idx2]

    # Leakage checks
    assert not (set(g_train.tolist()) & set(g_val.tolist())), "Leakage train∩val"
    assert not (set(g_train.tolist()) & set(g_test.tolist())), "Leakage train∩test"
    assert not (set(g_val.tolist()) & set(g_test.tolist())), "Leakage val∩test"
    print(f"  Leakage check OK — train dogs={len(set(g_train.tolist()))}, "
          f"val={len(set(g_val.tolist()))}, test={len(set(g_test.tolist()))}")

    C = len(present_channels)
    T = args.window_len
    meta_base = {
        "label2id": label2id,
        "id2label": id2label,
        "channel_names": present_channels,
        "channel0_physical": present_channels[0],
        "num_channels": C,
        "dataset_id": out_name,
        "sampling_hz": None,
        "window_len": T,
        "source_csv": args.source_csv,
        "split_policy": "subject_held_out",
    }

    _write_split(X_train, y_train, g_train, "train", out_root, meta_base, ts)
    _write_split(X_val, y_val, g_val, "val", out_root, meta_base, ts)
    _write_split(X_test, y_test, g_test, "test", out_root, meta_base, ts)

    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
