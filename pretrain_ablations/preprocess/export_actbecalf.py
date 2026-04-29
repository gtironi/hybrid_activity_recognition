"""Export AcTBeCalf long-format parquet to canonical .pt via sliding window."""

from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import hash_tensor, make_timestamp, repo_root, save_json

CHANNEL_NAMES = ["accX", "accY", "accZ"]


def build_windows(df, window_len, stride):
    all_windows, all_labels, all_groups = [], [], []
    for (calf_id, seg_id), grp in df.groupby(["calfId", "segId"], sort=False):
        grp = grp.reset_index(drop=True)
        data = grp[CHANNEL_NAMES].values.astype(np.float32)
        behaviours = grp["behaviour"].values
        N = len(data)
        for start in range(0, N - window_len + 1, stride):
            end = start + window_len
            lbls = behaviours[start:end]
            unique = np.unique(lbls)
            if len(unique) != 1:
                continue
            all_windows.append(data[start:end].T)
            all_labels.append(str(unique[0]))
            all_groups.append(int(calf_id))
    if not all_windows:
        return np.zeros((0, 3, window_len), dtype=np.float32), np.array([]), np.array([])
    return (np.array(all_windows, dtype=np.float32),
            np.array(all_labels), np.array(all_groups, dtype=np.int64))


def write_split(samples, labels, groups, source_indices, split, out_root, meta_base, ts, le):
    y = le.transform(labels).astype(np.int64)
    t_s = torch.tensor(samples, dtype=torch.float32)
    t_l = torch.tensor(y, dtype=torch.long)
    t_g = torch.tensor(groups, dtype=torch.long)
    id2label = {str(int(i)): cls for i, cls in enumerate(le.classes_)}
    cnt = Counter(y.tolist())
    class_dist = {id2label[str(k)]: v for k, v in sorted(cnt.items())}
    N, C, T = samples.shape
    meta = {**meta_base, "split": split, "export_timestamp": ts,
            "sha256_samples": hash_tensor(t_s), "id2label": id2label}
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": t_s, "labels": t_l, "groups": t_g, "meta": meta},
               out_root / f"{split}.pt")
    print(f"  [{split}] shape=({N},{C},{T}) calves={len(set(groups.tolist()))}")
    splits_dir = out_root / "splits"; splits_dir.mkdir(exist_ok=True)
    save_json({"indices": source_indices.tolist(), "class_distribution": class_dist,
               "group_ids": sorted(set(groups.tolist())), "n_samples": N},
              splits_dir / f"{split}_indices.json")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--window_len", type=int, default=75)
    p.add_argument("--stride", type=int, default=37)
    p.add_argument("--min_class_count", type=int, default=50)
    p.add_argument("--val_size", type=float, default=0.1)
    args = p.parse_args()

    root = repo_root()
    src_root = root / "dataset" / "processed" / "AcTBeCalf"
    out_root = root / "pretrain_ablations" / "processed" / "actbecalf_windowed"
    ts = make_timestamp()

    df_train = pd.read_parquet(src_root / "train.parquet")
    df_test = pd.read_parquet(src_root / "test.parquet")
    print(f"Train raw: {len(df_train)} | Test raw: {len(df_test)}")

    X_tr, y_tr, g_tr = build_windows(df_train, args.window_len, args.stride)
    X_te, y_te, g_te = build_windows(df_test, args.window_len, args.stride)
    idx_tr = np.arange(len(X_tr)); idx_te = np.arange(len(X_te))
    print(f"  Windows — train: {len(X_tr)} test: {len(X_te)}")

    train_counts = Counter(y_tr.tolist())
    rare = {cls for cls, cnt in train_counts.items() if cnt < args.min_class_count}
    if rare:
        print(f"  Drop rare classes: {sorted(rare)}")
        mask_tr = np.array([l not in rare for l in y_tr])
        mask_te = np.array([l not in rare for l in y_te])
        X_tr, y_tr, g_tr, idx_tr = X_tr[mask_tr], y_tr[mask_tr], g_tr[mask_tr], idx_tr[mask_tr]
        X_te, y_te, g_te, idx_te = X_te[mask_te], y_te[mask_te], g_te[mask_te], idx_te[mask_te]

    le = LabelEncoder(); le.fit(y_tr)
    known = set(le.classes_)
    mask_te = np.array([l in known for l in y_te])
    X_te, y_te, g_te, idx_te = X_te[mask_te], y_te[mask_te], g_te[mask_te], idx_te[mask_te]
    print(f"  Classes: {len(le.classes_)} | Train: {len(X_tr)} | Test: {len(X_te)}")
    save_json({cls: int(i) for i, cls in enumerate(le.classes_)}, out_root / "label2id.json")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=42)
    train_pos, val_pos = next(gss.split(X_tr, y_tr, groups=g_tr))
    X_val, y_val, g_val, idx_val = X_tr[val_pos], y_tr[val_pos], g_tr[val_pos], idx_tr[val_pos]
    X_tr, y_tr, g_tr, idx_tr = X_tr[train_pos], y_tr[train_pos], g_tr[train_pos], idx_tr[train_pos]
    assert not (set(g_tr.tolist()) & set(g_val.tolist())), "Leakage!"
    print(f"  Split — train calves={len(set(g_tr.tolist()))} val={len(set(g_val.tolist()))} test={len(set(g_te.tolist()))}")

    T = args.window_len
    meta_base = {"label2id": {cls: int(i) for i, cls in enumerate(le.classes_)},
                 "channel_names": CHANNEL_NAMES, "channel0_physical": "accX",
                 "num_channels": 3, "dataset_id": "actbecalf_windowed",
                 "sampling_hz": 25.0, "window_len": T, "split_policy": "group_split_calf_id"}
    write_split(X_tr, y_tr, g_tr, idx_tr, "train", out_root, meta_base, ts, le)
    write_split(X_val, y_val, g_val, idx_val, "val", out_root, meta_base, ts, le)
    write_split(X_te, y_te, g_te, idx_te, "test", out_root, meta_base, ts, le)
    print(f"Done: {out_root}")

if __name__ == "__main__":
    main()
