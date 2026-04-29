"""Export dog CSVs (each row = raw timestep) to canonical .pt via sliding window."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import hash_tensor, make_timestamp, repo_root, save_json

ACC_CHANNELS = ["ABack_x", "ABack_y", "ABack_z", "ANeck_x", "ANeck_y", "ANeck_z"]


def sliding_windows(df_group, channels, window_len, stride):
    data = df_group[channels].values.astype(np.float32)
    labels = df_group["label"].values if "label" in df_group.columns else None
    wins, win_labels = [], []
    for start in range(0, len(data) - window_len + 1, stride):
        end = start + window_len
        if labels is not None:
            unique = np.unique(labels[start:end])
            if len(unique) != 1:
                continue
            win_labels.append(str(unique[0]))
        else:
            win_labels.append("unknown")
        wins.append(data[start:end].T)
    return wins, win_labels


def write_split(samples, labels, groups, source_indices, split, out_root, meta_base, ts):
    N, C, T = samples.shape
    t_s = torch.tensor(samples, dtype=torch.float32)
    t_l = torch.tensor(labels, dtype=torch.long)
    t_g = torch.tensor(groups, dtype=torch.long)
    id2label: dict = meta_base["id2label"]
    cnt = Counter(labels.tolist())
    class_dist = {id2label.get(str(k), str(k)): v for k, v in sorted(cnt.items())}
    meta = {**meta_base, "split": split, "export_timestamp": ts,
            "sha256_samples": hash_tensor(t_s)}
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": t_s, "labels": t_l, "groups": t_g, "meta": meta},
               out_root / f"{split}.pt")
    print(f"  [{split}] shape={tuple(t_s.shape)} dogs={len(set(groups.tolist()))}")
    splits_dir = out_root / "splits"; splits_dir.mkdir(exist_ok=True)
    save_json({"indices": source_indices.tolist(), "class_distribution": class_dist,
               "group_ids": sorted(set(groups.tolist())), "n_samples": N},
              splits_dir / f"{split}_indices.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_csv", default="dog_w50.csv")
    p.add_argument("--window_len", type=int, default=10)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    args = p.parse_args()

    root = repo_root()
    csv_path = root / "data" / "dog" / args.source_csv
    source_tag = args.source_csv.replace(".csv", "").lstrip("dog_")
    out_name = f"dog_{source_tag}_w{args.window_len}"
    out_root = root / "pretrain_ablations" / "processed" / out_name
    ts = make_timestamp()

    df = pd.read_csv(csv_path)
    df = df.sort_values(["DogID", "TestNum"]).reset_index(drop=True)
    channels = [c for c in ACC_CHANNELS if c in df.columns]
    print(f"Source: {args.source_csv} | channels: {channels} | window={args.window_len} stride={args.stride}")

    all_windows, all_labels, all_dogs = [], [], []
    for (dog_id, test_num), grp in df.groupby(["DogID", "TestNum"]):
        wins, lbls = sliding_windows(grp.reset_index(drop=True), channels, args.window_len, args.stride)
        all_windows.extend(wins); all_labels.extend(lbls)
        all_dogs.extend([int(dog_id)] * len(wins))

    print(f"  Total windows: {len(all_windows)}")
    X = np.array(all_windows, dtype=np.float32)
    dog_ids = np.array(all_dogs, dtype=np.int64)
    all_idx = np.arange(len(X))

    le = LabelEncoder(); y = le.fit_transform(all_labels).astype(np.int64)
    label2id = {cls: int(i) for i, cls in enumerate(le.classes_)}
    id2label = {str(int(i)): cls for i, cls in enumerate(le.classes_)}
    save_json(label2id, out_root / "label2id.json")
    print(f"  {len(le.classes_)} classes")

    gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    tv_pos, te_pos = next(gss_test.split(X, y, groups=dog_ids))
    X_te=X[te_pos]; y_te=y[te_pos]; g_te=dog_ids[te_pos]; idx_te=all_idx[te_pos]
    X_tv=X[tv_pos]; y_tv=y[tv_pos]; g_tv=dog_ids[tv_pos]; idx_tv=all_idx[tv_pos]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=args.val_size/(1-args.test_size), random_state=42)
    tr_pos, vl_pos = next(gss_val.split(X_tv, y_tv, groups=g_tv))
    X_tr=X_tv[tr_pos]; y_tr=y_tv[tr_pos]; g_tr=g_tv[tr_pos]; idx_tr=idx_tv[tr_pos]
    X_vl=X_tv[vl_pos]; y_vl=y_tv[vl_pos]; g_vl=g_tv[vl_pos]; idx_vl=idx_tv[vl_pos]

    assert not (set(g_tr.tolist()) & set(g_vl.tolist())), "Leakage train/val"
    assert not (set(g_tr.tolist()) & set(g_te.tolist())), "Leakage train/test"
    print(f"  Split — train dogs={len(set(g_tr.tolist()))} val={len(set(g_vl.tolist()))} test={len(set(g_te.tolist()))}")

    C = len(channels)
    meta_base = {"label2id": label2id, "id2label": id2label, "channel_names": channels,
                 "channel0_physical": channels[0], "num_channels": C,
                 "dataset_id": out_name, "sampling_hz": None,
                 "window_len": args.window_len, "source_csv": args.source_csv,
                 "split_policy": "subject_held_out"}
    write_split(X_tr, y_tr, g_tr, idx_tr, "train", out_root, meta_base, ts)
    write_split(X_vl, y_vl, g_vl, idx_vl, "val", out_root, meta_base, ts)
    write_split(X_te, y_te, g_te, idx_te, "test", out_root, meta_base, ts)
    print(f"Done: {out_root}")

if __name__ == "__main__":
    main()
