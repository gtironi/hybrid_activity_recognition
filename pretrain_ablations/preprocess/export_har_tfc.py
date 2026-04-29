"""Export HAR from TFC .pt format to canonical pretrain_ablations .pt."""

from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import hash_tensor, make_timestamp, repo_root, save_json

CHANNEL_NAMES = ["body_acc_x", "body_acc_y", "body_acc_z"]
LABEL_NAMES = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"]


def export_split(split, src_root, out_root, ts):
    d = torch.load(src_root / f"{split}.pt", weights_only=False)
    samples = d["samples"]
    if not isinstance(samples, torch.Tensor):
        samples = torch.stack(list(samples))
    labels = d["labels"]
    if not isinstance(labels, torch.Tensor):
        labels = torch.stack(list(labels))
    if samples.ndim == 2:
        samples = samples.unsqueeze(1)
    # ensure (N,C,T) with C in dim1
    if samples.shape.index(min(samples.shape)) == 2:
        samples = samples.permute(0, 2, 1)
    samples = samples.float()
    labels = labels.long()
    N, C, T = samples.shape
    groups = torch.full((N,), -1, dtype=torch.long)

    # try load subject IDs
    raw_root = repo_root() / "data" / "HAR UCI"
    fname = "subject_train.txt" if split == "train" else "subject_test.txt"
    folder = "train" if split == "train" else "test"
    subj_path = raw_root / folder / fname
    if subj_path.exists():
        subjs = torch.tensor(list(map(int, subj_path.read_text().split())), dtype=torch.long)
        if len(subjs) == N:
            groups = subjs

    label2id = {n: i for i, n in enumerate(LABEL_NAMES)}
    id2label = {str(i): n for i, n in enumerate(LABEL_NAMES)}
    cnt = Counter(labels.tolist())
    class_dist = {LABEL_NAMES[k]: v for k, v in sorted(cnt.items())}

    meta = {
        "label2id": label2id, "id2label": id2label,
        "channel_names": CHANNEL_NAMES, "channel0_physical": "body_acc_x",
        "num_channels": C, "dataset_id": "har_tfc",
        "sampling_hz": 50.0, "window_len": T,
        "split": split, "split_policy": "subject_held_out",
        "export_timestamp": ts, "sha256_samples": hash_tensor(samples),
    }
    canonical = {"samples": samples, "labels": labels, "groups": groups, "meta": meta}
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save(canonical, out_root / f"{split}.pt")
    print(f"  [{split}] shape={tuple(samples.shape)}")
    splits_dir = out_root / "splits"; splits_dir.mkdir(exist_ok=True)
    save_json({"indices": list(range(N)), "class_distribution": class_dist,
               "group_ids": sorted(set(groups.tolist())), "n_samples": N},
              splits_dir / f"{split}_indices.json")


def main():
    root = repo_root()
    src_root = root / "data" / "processed" / "HAR"
    out_root = root / "pretrain_ablations" / "processed" / "har_tfc"
    ts = make_timestamp()
    label2id = {n: i for i, n in enumerate(LABEL_NAMES)}
    save_json(label2id, out_root / "label2id.json")
    for split in ["train", "val", "test"]:
        if (src_root / f"{split}.pt").exists():
            export_split(split, src_root, out_root, ts)
    print(f"Done: {out_root}")

if __name__ == "__main__":
    main()
