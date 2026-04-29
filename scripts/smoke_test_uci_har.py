#!/usr/bin/env python3
"""
Smoke test for UCI HAR windowed parquets.

Checks:
  - Files exist and are non-empty
  - Required columns present
  - acc_x/acc_y/acc_z are lists of length 128
  - No NaN in label or subject columns
  - Train/test subjects are disjoint
  - All test labels appear in train labels
  - TSFEL feature columns are identical in train and test
  - Basic shape sanity (windows count, class distribution)

Usage:
  python scripts/smoke_test_uci_har.py
  python scripts/smoke_test_uci_har.py --processed-dir dataset/processed/UCI_HAR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLS = {"dateTime", "calf_id", "acc_x", "acc_y", "acc_z", "label"}
UCI_WINDOW_SIZE = 128
EXPECTED_CLASSES = {
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
}


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"  OK:   {msg}")


def check(cond: bool, ok_msg: str, fail_msg: str) -> bool:
    if cond:
        ok(ok_msg)
    else:
        fail(fail_msg)
    return cond


def smoke_test(processed_dir: Path) -> bool:
    train_path = processed_dir / "windowed_train.parquet"
    test_path = processed_dir / "windowed_test.parquet"
    manifest_path = processed_dir / "tsfel_feature_manifest.json"

    passed = True

    # ── file existence ────────────────────────────────────────────────────────
    print("\n[1] File existence")
    for p in (train_path, test_path, manifest_path):
        if not check(p.exists(), f"exists: {p.name}", f"missing: {p}"):
            passed = False

    if not (train_path.exists() and test_path.exists()):
        print("\nCannot continue without parquet files.")
        return False

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    # ── non-empty ─────────────────────────────────────────────────────────────
    print("\n[2] Non-empty")
    check(len(train) > 0, f"train rows={len(train):,}", "train is empty")
    check(len(test) > 0, f"test rows={len(test):,}", "test is empty")

    # ── required columns ──────────────────────────────────────────────────────
    print("\n[3] Required columns")
    for split, df in (("train", train), ("test", test)):
        missing = REQUIRED_COLS - set(df.columns)
        if not check(not missing, f"{split}: all required cols present", f"{split}: missing {missing}"):
            passed = False

    # ── TSFEL cols match ──────────────────────────────────────────────────────
    print("\n[4] TSFEL columns consistent train/test")
    meta_cols = list(REQUIRED_COLS)
    tsfel_train = set(train.columns) - set(meta_cols)
    tsfel_test = set(test.columns) - set(meta_cols)
    only_train = tsfel_train - tsfel_test
    only_test = tsfel_test - tsfel_train
    if not check(not only_train and not only_test,
                 f"TSFEL cols match ({len(tsfel_train)} cols)",
                 f"mismatch: only_train={len(only_train)} only_test={len(only_test)}"):
        passed = False

    # ── window size ───────────────────────────────────────────────────────────
    print("\n[5] Window size (acc lists = 128 samples)")
    for split, df in (("train", train), ("test", test)):
        if "acc_x" in df.columns and len(df) > 0:
            sample_len = len(df["acc_x"].iloc[0])
            if not check(sample_len == UCI_WINDOW_SIZE,
                         f"{split}: acc_x len={sample_len}",
                         f"{split}: acc_x len={sample_len} != {UCI_WINDOW_SIZE}"):
                passed = False

    # ── no NaN in key cols ────────────────────────────────────────────────────
    print("\n[6] No NaN in label / calf_id")
    for split, df in (("train", train), ("test", test)):
        for col in ("label", "calf_id"):
            if col in df.columns:
                n_nan = df[col].isna().sum()
                if not check(n_nan == 0, f"{split}.{col}: no NaN", f"{split}.{col}: {n_nan} NaN"):
                    passed = False

    # ── subject disjointness ──────────────────────────────────────────────────
    print("\n[7] Train/test subjects disjoint")
    train_subj = set(train["calf_id"].unique()) if "calf_id" in train.columns else set()
    test_subj = set(test["calf_id"].unique()) if "calf_id" in test.columns else set()
    overlap = train_subj & test_subj
    if not check(not overlap,
                 f"disjoint (train={len(train_subj)}, test={len(test_subj)})",
                 f"overlap subjects: {sorted(overlap)}"):
        passed = False

    # ── label alignment ───────────────────────────────────────────────────────
    print("\n[8] All test labels present in train")
    if "label" in train.columns and "label" in test.columns:
        train_labels = set(train["label"].unique())
        test_labels = set(test["label"].unique())
        unseen = test_labels - train_labels
        if not check(not unseen,
                     f"all test labels in train ({sorted(test_labels)})",
                     f"test labels not in train: {unseen}"):
            passed = False

    # ── expected UCI classes ──────────────────────────────────────────────────
    print("\n[9] Expected UCI classes present in train")
    if "label" in train.columns:
        found = set(train["label"].unique())
        missing_cls = EXPECTED_CLASSES - found
        extra_cls = found - EXPECTED_CLASSES
        check(not missing_cls,
              f"all 6 UCI classes present",
              f"missing classes: {missing_cls}")
        if extra_cls:
            ok(f"extra classes (not an error): {extra_cls}")

    # ── class distribution summary ────────────────────────────────────────────
    print("\n[10] Class distribution")
    for split, df in (("train", train), ("test", test)):
        if "label" in df.columns:
            dist = df["label"].value_counts().to_dict()
            print(f"  {split}:")
            for cls, cnt in sorted(dist.items()):
                print(f"    {cls:<25} {cnt:>6,}")

    # ── NaN in TSFEL features ─────────────────────────────────────────────────
    print("\n[11] NaN in TSFEL feature columns")
    for split, df in (("train", train), ("test", test)):
        feat_cols = [c for c in df.columns if c not in REQUIRED_COLS]
        if feat_cols:
            n_nan = df[feat_cols].isna().sum().sum()
            check(n_nan == 0, f"{split}: no NaN in {len(feat_cols)} TSFEL cols",
                  f"{split}: {n_nan:,} NaN across TSFEL cols")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if passed:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED — see FAIL lines above")
    return passed


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test UCI HAR windowed parquets")
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("dataset/processed/UCI_HAR"),
    )
    args = p.parse_args()

    ok_flag = smoke_test(args.processed_dir)
    sys.exit(0 if ok_flag else 1)


if __name__ == "__main__":
    main()
