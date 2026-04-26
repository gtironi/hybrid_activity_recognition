"""Export AcTBeCalf long-format parquet to canonical .pt via sliding window.

Source: dataset/processed/AcTBeCalf/{train,test}.parquet
  Columns: dateTime, calfId, accX, accY, accZ, behaviour, segId

Groups per window by (calfId, segId). Only 100%-pure windows kept.
Group split by calfId: existing train/test preserved; val carved from train.

Usage:
    python export_actbecalf.py [--window_len 75] [--stride 37] [--min_class_count 50]
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

CHANNEL_NAMES = ["accX", "accY", "accZ"]


def build_windows(df: pd.DataFrame, window_len: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding window over (calfId, segId) groups.
    Returns:
        samples: (N, 3, window_len) float32
        labels:  (N,) str
        groups:  (N,) int  — calf_id encoded as int
    """
    all_windows, all_labels, all_groups = [], [], []

    for (calf_id, seg_id), grp in df.groupby(["calfId", "segId"], sort=False):
        grp = grp.reset_index(drop=True)
        data = grp[CHANNEL_NAMES].values.astype(np.float32)  # (T, 3)
        behaviours = grp["behaviour"].values
        N = len(data)

        for start in range(0, N - window_len + 1, stride):
            end = start + window_len
            lbls = behaviours[start:end]
            unique = np.unique(lbls)
            if len(unique) != 1:
                continue  # mixed label — skip
            all_windows.append(data[start:end].T)  # (3, window_len)
            all_labels.append(str(unique[0]))
            all_groups.append(int(calf_id))

    if not all_windows:
        return np.zeros((0, 3, window_len), dtype=np.float32), np.array([]), np.array([])

    return (
        np.array(all_windows, dtype=np.float32),
        np.array(all_labels),
        np.array(all_groups, dtype=np.int64),
    )


def _write_split(
    samples: np.ndarray, labels: np.ndarray, groups: np.ndarray,
    split: str, out_root: Path, meta_base: dict, ts: str, le: LabelEncoder,
) -> None:
    y = le.transform(labels).astype(np.int64)
    t_samples = torch.tensor(samples, dtype=torch.float32)
    t_labels = torch.tensor(y, dtype=torch.long)
    t_groups = torch.tensor(groups, dtype=torch.long)

    id2label = {str(int(i)): cls for i, cls in enumerate(le.classes_)}
    cnt = Counter(y.tolist())
    class_dist = {id2label[str(k)]: v for k, v in sorted(cnt.items())}

    N, C, T = samples.shape
    meta = {**meta_base, "split": split, "export_timestamp": ts,
            "sha256_samples": hash_tensor(t_samples),
            "id2label": id2label}
    canonical = {"samples": t_samples, "labels": t_labels, "groups": t_groups, "meta": meta}

    out_root.mkdir(parents=True, exist_ok=True)
    torch.save(canonical, out_root / f"{split}.pt")
    print(f"  [{split}] shape=({N},{C},{T}) windows={N} calves={len(set(groups.tolist()))}")

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
    parser.add_argument("--window_len", type=int, default=75)
    parser.add_argument("--stride", type=int, default=37)
    parser.add_argument("--min_class_count", type=int, default=50,
                        help="Drop classes with fewer windows than this in train set")
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()

    registry = load_registry()
    root = repo_root()
    src_root = root / "dataset" / "processed" / "AcTBeCalf"
    out_root = root / "pretrain_ablations" / "processed" / "actbecalf_windowed"
    ts = make_timestamp()

    print(f"Loading parquets (window_len={args.window_len}, stride={args.stride})...")
    df_train = pd.read_parquet(src_root / "train.parquet")
    df_test = pd.read_parquet(src_root / "test.parquet")
    print(f"  Train raw: {len(df_train)} rows | Test raw: {len(df_test)} rows")

    print("Building windows from train...")
    X_tr, y_tr, g_tr = build_windows(df_train, args.window_len, args.stride)
    print(f"  Train windows: {len(X_tr)}")

    print("Building windows from test...")
    X_te, y_te, g_te = build_windows(df_test, args.window_len, args.stride)
    print(f"  Test windows: {len(X_te)}")

    # Drop rare classes (< min_class_count in train)
    train_counts = Counter(y_tr.tolist())
    rare = {cls for cls, cnt in train_counts.items() if cnt < args.min_class_count}
    if rare:
        print(f"  Dropping rare classes (< {args.min_class_count}): {sorted(rare)}")
        mask_tr = np.array([lbl not in rare for lbl in y_tr])
        mask_te = np.array([lbl not in rare for lbl in y_te])
        X_tr, y_tr, g_tr = X_tr[mask_tr], y_tr[mask_tr], g_tr[mask_tr]
        X_te, y_te, g_te = X_te[mask_te], y_te[mask_te], g_te[mask_te]

    # Encode labels — fit on train
    le = LabelEncoder()
    le.fit(y_tr)
    known = set(le.classes_)
    mask_te_known = np.array([lbl in known for lbl in y_te])
    X_te, y_te, g_te = X_te[mask_te_known], y_te[mask_te_known], g_te[mask_te_known]

    print(f"  Classes: {len(le.classes_)} | Train: {len(X_tr)} | Test: {len(X_te)}")

    label2id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    id2label = {str(int(i)): cls for i, cls in enumerate(le.classes_)}
    save_json(label2id, out_root / "label2id.json")

    # Carve val from train by calfId
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=42)
    train_idx, val_idx = next(gss.split(X_tr, y_tr, groups=g_tr))

    X_val, y_val, g_val = X_tr[val_idx], y_tr[val_idx], g_tr[val_idx]
    X_tr, y_tr, g_tr = X_tr[train_idx], y_tr[train_idx], g_tr[train_idx]

    # Leakage check
    assert not (set(g_tr.tolist()) & set(g_val.tolist())), "Leakage train∩val"
    print(f"  Leakage OK — train calves={len(set(g_tr.tolist()))}, "
          f"val={len(set(g_val.tolist()))}, test={len(set(g_te.tolist()))}")

    meta_base = {
        "label2id": label2id,
        "id2label": id2label,
        "channel_names": CHANNEL_NAMES,
        "channel0_physical": "accX",
        "num_channels": 3,
        "dataset_id": "actbecalf_windowed",
        "sampling_hz": 25.0,
        "window_len": args.window_len,
        "split_policy": "group_split_calf_id",
    }

    _write_split(X_tr, y_tr, g_tr, "train", out_root, meta_base, ts, le)
    _write_split(X_val, y_val, g_val, "val", out_root, meta_base, ts, le)
    _write_split(X_te, y_te, g_te, "test", out_root, meta_base, ts, le)

    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
