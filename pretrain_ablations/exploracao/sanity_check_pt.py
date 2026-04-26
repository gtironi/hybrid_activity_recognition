"""Phase 3 QA: numeric sanity checks on exported canonical .pt files.

Usage:
    python sanity_check_pt.py --dataset har_tfc
    python sanity_check_pt.py --dataset actbecalf_windowed
    python sanity_check_pt.py --dataset dog_w50_w10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from utils import hash_tensor, load_registry, make_timestamp, repo_root, save_json


def check_split(pt_path: Path, split: str) -> dict:
    print(f"\n  [{split}] {pt_path}")
    if not pt_path.exists():
        print(f"    MISSING: {pt_path}")
        return {"split": split, "status": "missing"}

    d = torch.load(pt_path, weights_only=False)
    s = d["samples"]
    l = d["labels"]
    g = d["groups"]
    meta = d.get("meta", {})

    issues = []

    # Shape checks
    assert s.ndim == 3, f"samples must be 3D, got {s.ndim}D"
    N, C, T = s.shape
    print(f"    samples.shape = ({N}, {C}, {T})")
    print(f"    samples.dtype = {s.dtype}")
    print(f"    labels.shape  = {tuple(l.shape)}, dtype={l.dtype}")
    print(f"    groups.shape  = {tuple(g.shape)}, dtype={g.dtype}")

    if s.dtype != torch.float32:
        issues.append(f"samples.dtype={s.dtype} (expected float32)")
    if l.dtype != torch.long:
        issues.append(f"labels.dtype={l.dtype} (expected int64/long)")
    if len(l) != N:
        issues.append(f"labels length {len(l)} != samples N={N}")

    # NaN / Inf
    nan_count = s.isnan().sum().item()
    inf_count = s.isinf().sum().item()
    print(f"    NaN={nan_count} Inf={inf_count}")
    if nan_count > 0:
        issues.append(f"NaN in samples: {nan_count}")
    if inf_count > 0:
        issues.append(f"Inf in samples: {inf_count}")

    # Value range
    s_min = s.min().item()
    s_max = s.max().item()
    s_mean = s.float().mean().item()
    s_std = s.float().std().item()
    print(f"    min={s_min:.4f} max={s_max:.4f} mean={s_mean:.4f} std={s_std:.4f}")

    # Labels
    unique_labels = sorted(l.unique().tolist())
    print(f"    labels unique = {unique_labels}")
    from collections import Counter
    cnt = Counter(l.tolist())
    id2label = meta.get("id2label", {})
    class_dist = {id2label.get(str(k), str(k)): v for k, v in sorted(cnt.items())}
    print(f"    class distribution: {class_dist}")

    # Groups
    non_minus1 = g[g != -1]
    unique_groups = sorted(set(non_minus1.tolist())) if len(non_minus1) > 0 else []
    print(f"    groups unique = {len(unique_groups)} (first 5: {unique_groups[:5]})")

    # Meta consistency
    if "num_channels" in meta and meta["num_channels"] != C:
        issues.append(f"meta.num_channels={meta['num_channels']} != actual C={C}")
    if "window_len" in meta and meta["window_len"] != T:
        issues.append(f"meta.window_len={meta['window_len']} != actual T={T}")

    # SHA256 check
    if "sha256_samples" in meta:
        computed = hash_tensor(s.contiguous())
        if computed == meta["sha256_samples"]:
            print(f"    SHA256: OK")
        else:
            issues.append(f"SHA256 mismatch! stored={meta['sha256_samples'][:16]}... computed={computed[:16]}...")

    if issues:
        print(f"    ISSUES: {issues}")
    else:
        print(f"    All checks PASSED")

    return {
        "split": split, "status": "ok" if not issues else "issues",
        "shape": [N, C, T], "nan": nan_count, "inf": inf_count,
        "min": s_min, "max": s_max, "mean": s_mean, "std": s_std,
        "num_classes": len(unique_labels), "class_distribution": class_dist,
        "num_groups": len(unique_groups), "issues": issues,
    }


def check_cross_split_leakage(processed_root: Path) -> None:
    """Check no group_id appears in more than one split."""
    split_groups: dict[str, set] = {}
    for split in ["train", "val", "test"]:
        pt = processed_root / f"{split}.pt"
        if not pt.exists():
            continue
        d = torch.load(pt, weights_only=False)
        groups = d["groups"].tolist()
        split_groups[split] = set(g for g in groups if g != -1)

    splits = list(split_groups.keys())
    violations = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_groups[splits[i]] & split_groups[splits[j]]
            if overlap:
                violations.append((splits[i], splits[j], overlap))

    if violations:
        for s1, s2, ov in violations:
            print(f"  !! LEAKAGE: {s1} ∩ {s2} = {len(ov)} groups")
    else:
        counts = {s: len(v) for s, v in split_groups.items()}
        print(f"  Cross-split leakage: NONE ({counts})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check exported .pt files")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed_root", default=None,
                        help="Override path to processed/{dataset}/ directory")
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    ts = make_timestamp()

    if args.processed_root:
        processed_root = Path(args.processed_root)
    else:
        info = registry.get(args.dataset, {})
        pr = info.get("processed_root")
        if pr:
            processed_root = repo_root() / pr
        else:
            processed_root = repo_root() / "pretrain_ablations" / "processed" / args.dataset

    print(f"Checking: {processed_root}")
    if not processed_root.exists():
        sys.exit(f"Processed directory not found: {processed_root}\nRun the export script first.")

    results = {}
    for split in ["train", "val", "test"]:
        pt_path = processed_root / f"{split}.pt"
        results[split] = check_split(pt_path, split)

    print(f"\n  Cross-split leakage check:")
    check_cross_split_leakage(processed_root)

    # Summary
    all_ok = all(r.get("status") == "ok" for r in results.values() if r.get("status") != "missing")
    print(f"\n{'='*50}")
    print(f"Dataset: {args.dataset}")
    print(f"Overall: {'ALL PASSED' if all_ok else 'ISSUES FOUND'}")

    out_path = processed_root / f"sanity_{ts}.json"
    save_json(results, out_path)
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
