#!/usr/bin/env python3
"""
UCI HAR (pre-windowed 128-sample, 50 Hz) → windowed parquet + TSFEL.

UCI HAR already provides fixed-width windows (128 readings, 50% overlap, 50 Hz) with
train/test split that is subject-disjoint by construction. So we skip
`dataset_processing.py` and `create_windowed_dataframe`: each row of
`Inertial Signals/total_acc_{x,y,z}_{split}.txt` is already one window.

Mode discover (no --feature-manifest-in): sample + RF picks top-N TSFEL columns; writes manifest.
Mode apply (--feature-manifest-in): only extract manifest columns (no leakage on test).

Examples:

  # 1) Train (discover)
  python scripts/uci_har_to_windowed.py \\
    --root "data/HAR UCI" --split train \\
    --output dataset/processed/UCI_HAR/windowed_train.parquet \\
    --feature-manifest-out dataset/processed/UCI_HAR/tsfel_feature_manifest.json

  # 2) Test (apply)
  python scripts/uci_har_to_windowed.py \\
    --root "data/HAR UCI" --split test \\
    --output dataset/processed/UCI_HAR/windowed_test.parquet \\
    --feature-manifest-in dataset/processed/UCI_HAR/tsfel_feature_manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_windowed_parquet import (
    build_manifest,
    discover_top_features,
    extract_tsfel_batched,
    load_manifest,
)

UCI_FS = 50
UCI_WINDOW_SIZE = 128
UCI_OVERLAP = 0.5


def _read_signal_matrix(path: Path) -> np.ndarray:
    """UCI inertial files: whitespace-separated, 128 floats per row."""
    return np.loadtxt(path, dtype=np.float32)


def _read_activity_map(root: Path) -> dict[int, str]:
    df = pd.read_csv(
        root / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"]
    )
    return dict(zip(df["id"].astype(int), df["name"].astype(str)))


def load_uci_split(root: Path, split: str) -> pd.DataFrame:
    """
    Build dataframe with one row per pre-defined window, schema compatible with
    extract_tsfel_batched: acc_x/acc_y/acc_z lists, label, calf_id, dateTime.
    """
    if split not in {"train", "test"}:
        raise SystemExit(f"--split must be train|test, got {split!r}")

    sig_dir = root / split / "Inertial Signals"
    suffix = f"_{split}.txt"
    ax = _read_signal_matrix(sig_dir / f"total_acc_x{suffix}")
    ay = _read_signal_matrix(sig_dir / f"total_acc_y{suffix}")
    az = _read_signal_matrix(sig_dir / f"total_acc_z{suffix}")

    if not (ax.shape == ay.shape == az.shape):
        raise SystemExit(f"Signal shape mismatch: {ax.shape} {ay.shape} {az.shape}")
    if ax.shape[1] != UCI_WINDOW_SIZE:
        raise SystemExit(f"Expected {UCI_WINDOW_SIZE} samples per window, got {ax.shape[1]}")

    y = np.loadtxt(root / split / f"y{suffix}", dtype=np.int64)
    subj = np.loadtxt(root / split / f"subject{suffix}", dtype=np.int64)
    if not (len(y) == len(subj) == ax.shape[0]):
        raise SystemExit(
            f"Row count mismatch: signals={ax.shape[0]} y={len(y)} subject={len(subj)}"
        )

    label_map = _read_activity_map(root)
    labels = pd.Series(y).map(label_map).astype(str)

    df = pd.DataFrame(
        {
            "dateTime": np.arange(ax.shape[0], dtype=np.int64),
            "calf_id": subj.astype(np.int64),
            "acc_x": list(ax),
            "acc_y": list(ay),
            "acc_z": list(az),
            "label": labels.values,
        }
    )
    df["acc_x"] = df["acc_x"].apply(lambda a: a.tolist())
    df["acc_y"] = df["acc_y"].apply(lambda a: a.tolist())
    df["acc_z"] = df["acc_z"].apply(lambda a: a.tolist())

    print(
        f"UCI HAR {split}: {len(df):,} windows | subjects={df['calf_id'].nunique()} "
        f"| classes={df['label'].nunique()}"
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="UCI HAR → windowed parquet + TSFEL")
    p.add_argument("--root", type=Path, default=Path("data/HAR UCI"))
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature-manifest-out", type=Path, default=None)
    p.add_argument("--feature-manifest-in", type=Path, default=None)
    p.add_argument("--top-n", type=int, default=75)
    p.add_argument("--sample-size", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=5000)
    args = p.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Root not found: {args.root}")

    apply_mode = args.feature_manifest_in is not None
    if apply_mode and not args.feature_manifest_in.is_file():
        raise SystemExit(f"Manifest not found: {args.feature_manifest_in}")
    if apply_mode and args.feature_manifest_out is not None:
        print("Warn: apply mode ignores --feature-manifest-out.")

    df_main = load_uci_split(args.root, args.split)

    if apply_mode:
        manifest = load_manifest(args.feature_manifest_in)
        top_names = manifest["top_feature_names"]
    else:
        top_names = discover_top_features(
            df_main,
            top_n=args.top_n,
            sample_size=args.sample_size,
            fs=UCI_FS,
        )
        if not args.feature_manifest_out:
            print("Warn: no --feature-manifest-out; cannot replicate features on test.")
        else:
            man = build_manifest(
                top_names,
                window_size=UCI_WINDOW_SIZE,
                overlap=UCI_OVERLAP,
                purity_threshold=1.0,
                fs=UCI_FS,
                top_n=args.top_n,
                sample_size=args.sample_size,
                group_by=["calf_id"],
                label_column="label",
                acc_x="accX",
                acc_y="accY",
                acc_z="accZ",
                time_column="dateTime",
            )
            args.feature_manifest_out.parent.mkdir(parents=True, exist_ok=True)
            import json

            with open(args.feature_manifest_out, "w", encoding="utf-8") as f:
                json.dump(man, f, indent=2)
            print(f"Manifest: {args.feature_manifest_out.resolve()}")

    df_feat = extract_tsfel_batched(
        df_main,
        top_names,
        fs=UCI_FS,
        batch_size=args.batch_size,
    )
    df_main = df_main.reset_index(drop=True)
    df_final = pd.concat([df_main, df_feat], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(args.output, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved: {args.output.resolve()} shape={df_final.shape}")


if __name__ == "__main__":
    main()
