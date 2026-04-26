"""Export HAR UCI from TFC .pt format to canonical pretrain_ablations .pt format.

Source: data/processed/HAR/{train,val,test}.pt
Output: pretrain_ablations/processed/har_tfc/{train,val,test}.pt

All 9 channels preserved. Subject IDs loaded from raw UCI txt files for groups tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from utils import hash_tensor, load_registry, make_timestamp, repo_root, save_json

CHANNEL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

LABEL_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def _load_subjects(raw_root: Path, split: str) -> np.ndarray | None:
    fname = "subject_train.txt" if split == "train" else "subject_test.txt"
    p = raw_root / split / fname
    if not p.exists():
        # val split has no subject file in UCI dataset
        return None
    subjects = np.loadtxt(p, dtype=int)
    return subjects


def export_split(split: str, src_root: Path, raw_root: Path, out_root: Path, ts: str) -> None:
    pt_path = src_root / f"{split}.pt"
    print(f"  Loading {pt_path} ...")
    d = torch.load(pt_path, weights_only=False)

    samples = d["samples"]
    if not isinstance(samples, torch.Tensor):
        samples = torch.stack(list(samples))
    labels = d["labels"]
    if not isinstance(labels, torch.Tensor):
        labels = torch.stack(list(labels))

    # Ensure 3D (N, C, T)
    if samples.ndim == 2:
        samples = samples.unsqueeze(1)
    if samples.shape.index(min(samples.shape)) == 2:
        samples = samples.permute(0, 2, 1)

    samples = samples.float()
    labels = labels.long()

    N, C, T = samples.shape
    print(f"    shape={tuple(samples.shape)} labels={tuple(labels.shape)} unique={sorted(labels.unique().tolist())}")

    # Load subject IDs (groups)
    subjects = _load_subjects(raw_root, split)
    if subjects is not None and len(subjects) == N:
        groups = torch.tensor(subjects, dtype=torch.long)
    else:
        groups = torch.full((N,), -1, dtype=torch.long)
        if split not in ("val",):
            print(f"    Warning: subject file not found for {split}; groups=-1")

    # label2id / id2label
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    id2label = {str(i): name for i, name in enumerate(LABEL_NAMES)}

    # class distribution
    from collections import Counter
    cnt = Counter(labels.tolist())
    class_dist = {LABEL_NAMES[k]: v for k, v in sorted(cnt.items())}

    # Build canonical dict
    meta = {
        "label2id": label2id,
        "id2label": id2label,
        "channel_names": CHANNEL_NAMES,
        "channel0_physical": "body_acc_x",
        "num_channels": C,
        "dataset_id": "har_tfc",
        "sampling_hz": 50.0,
        "window_len": T,
        "split": split,
        "split_policy": "subject_held_out",
        "export_timestamp": ts,
        "sha256_samples": hash_tensor(samples),
    }
    canonical = {"samples": samples, "labels": labels, "groups": groups, "meta": meta}

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{split}.pt"
    torch.save(canonical, out_path)
    print(f"    saved: {out_path}")

    # splits JSON
    splits_dir = out_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    indices = list(range(N))
    split_json = {
        "indices": indices,
        "class_distribution": class_dist,
        "group_ids": sorted(set(groups.tolist())),
        "n_samples": N,
    }
    save_json(split_json, splits_dir / f"{split}_indices.json")
    print(f"    splits saved: {splits_dir / f'{split}_indices.json'}")


def main() -> None:
    registry = load_registry()
    info = registry["har_tfc"]
    root = repo_root()
    src_root = root / info["root"]
    raw_root = root / info["raw_root"]
    out_root = root / "pretrain_ablations" / "processed" / "har_tfc"
    ts = make_timestamp()

    for split in ["train", "val", "test"]:
        if not (src_root / f"{split}.pt").exists():
            print(f"  Skipping {split} (file not found)")
            continue
        export_split(split, src_root, raw_root, out_root, ts)

    # Save label2id.json at root level
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    save_json(label2id, out_root / "label2id.json")
    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
