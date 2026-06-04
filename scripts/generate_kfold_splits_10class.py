#!/usr/bin/env python3
"""
Generate K subject-level folds for the 10-class kfold experiment.

Two calves are anchored outside the fold pool:
  - TRAIN_ANCHOR (default 1306): always in train — richest Play calf,
    guarantees Play never falls below the proportion filter in any fold.
  - TEST_ANCHOR  (default 1357): always in test  — high Play proportion,
    guarantees Play is always observable at evaluation time.

The remaining (N-2) subjects are partitioned into K folds normally.
For each fold k:
  train = TRAIN_ANCHOR  + all non-test-fold subjects
  test  = TEST_ANCHOR   + fold_k subjects

Canonical labels are applied (full 20-class map); label remapping to 10
classes happens later at feature-extraction time via --remap-labels.

A lower min_train_proportion (default 0.005) keeps Play from being filtered
even when fold_k holds several Play-rich calves.

Usage:
  python scripts/generate_kfold_splits_10class.py \
      --out-dir dataset/processed/kfold_paper10c_w125 \
      --n-folds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import genSplit
from dataset_processing import (
    DEFAULT_BEHAVIOR_COLUMN,
    _subject_behavior_wide,
    apply_canonical_behavior_labels,
    build_split_report,
    filter_behaviors_below_min_train_proportion,
    filter_test_behaviors_to_train,
    split_subject_list,
)

DEFAULT_TRAIN_ANCHOR = 1306  # richest Play calf — always in train
DEFAULT_TEST_ANCHOR  = 1357  # high Play proportion — always in test
DEFAULT_MIN_PROP     = 0.005  # lower than default 0.01 to keep Play safe


def main() -> None:
    p = argparse.ArgumentParser(
        description="K-fold 10-class splits with anchored Play calves"
    )
    p.add_argument("--csv",      type=Path, default=Path("dataset/AcTBeCalf.csv"))
    p.add_argument("--out-dir",  type=Path, default=Path("dataset/processed/kfold_paper10c_w125"))
    p.add_argument("--n-folds",  type=int,  default=5)
    p.add_argument("--subject-column",  default="calfId")
    p.add_argument("--behavior-column", default=DEFAULT_BEHAVIOR_COLUMN)
    p.add_argument("--train-anchor", type=int, default=DEFAULT_TRAIN_ANCHOR,
                   help="Subject ID always assigned to train.")
    p.add_argument("--test-anchor",  type=int, default=DEFAULT_TEST_ANCHOR,
                   help="Subject ID always assigned to test.")
    p.add_argument("--min-train-proportion-per-behavior", type=float,
                   default=DEFAULT_MIN_PROP)
    args = p.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.csv} ...")
    df = pd.read_csv(args.csv)
    label_meta = apply_canonical_behavior_labels(df, args.behavior_column)

    all_subjects = sorted(df[args.subject_column].unique().tolist(), key=str)

    train_anchor = str(args.train_anchor)
    test_anchor  = str(args.test_anchor)

    for anchor, role in [(train_anchor, "train-anchor"), (test_anchor, "test-anchor")]:
        if anchor not in [str(s) for s in all_subjects]:
            raise SystemExit(f"{role} subject {anchor} not found in data.")
    if train_anchor == test_anchor:
        raise SystemExit("train-anchor and test-anchor must be different subjects.")

    pool = [s for s in all_subjects if str(s) not in (train_anchor, test_anchor)]
    print(f"Anchors  → train: {train_anchor}, test: {test_anchor}")
    print(f"Fold pool: {len(pool)} subjects  (total: {len(all_subjects)})")

    wide = _subject_behavior_wide(df, args.subject_column, args.behavior_column)
    pool_wide = wide[wide["subject_id"].isin([str(s) for s in pool] + [int(s) for s in pool])]

    folds = genSplit.partition_subjects_into_folds(pool, args.n_folds, pool_wide)

    assignments = {f"fold_{k}": sorted(folds[k], key=str) for k in range(args.n_folds)}
    assignments_path = args.out_dir / "fold_assignments.json"
    assignments_path.write_text(
        json.dumps(
            {
                "n_folds": args.n_folds,
                "n_subjects_pool": len(pool),
                "train_anchor": train_anchor,
                "test_anchor":  test_anchor,
                "subject_column":  args.subject_column,
                "behavior_column": args.behavior_column,
                "test_subjects_per_fold": assignments,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"Fold assignments: {assignments_path}")

    # Pass 1: discover consistent class set across all folds
    fold_data: list[dict] = []
    surviving_sets: list[set] = []

    for k in range(args.n_folds):
        test_pool_subjects = [str(s) for s in folds[k]]
        test_subjects_all  = test_pool_subjects + [test_anchor]
        train_subjects_all = (
            [str(s) for s in pool if str(s) not in test_pool_subjects]
            + [train_anchor]
        )

        train, _, _ = split_subject_list(
            df, subject_column=args.subject_column,
            test_subjects=test_subjects_all,
        )
        _, test, method = split_subject_list(
            df, subject_column=args.subject_column,
            test_subjects=test_subjects_all,
        )

        train, test, insuf = filter_behaviors_below_min_train_proportion(
            train, test,
            behavior_column=args.behavior_column,
            min_train_proportion=args.min_train_proportion_per_behavior,
        )
        surviving = set(train[args.behavior_column].astype(str).unique())
        surviving_sets.append(surviving)
        fold_data.append({"train": train, "test": test,
                          "method": method, "insufficient": insuf,
                          "test_subjects": test_subjects_all})

    consistent_classes = set.intersection(*surviving_sets)
    print(f"\nConsistent classes ({len(consistent_classes)}):")
    for c in sorted(consistent_classes):
        print(f"  - {c}")
    dropped = sorted(set.union(*surviving_sets) - consistent_classes)
    if dropped:
        print(f"Dropped for consistency: {dropped}")

    (args.out_dir / "consistent_classes.json").write_text(
        json.dumps({"consistent_classes": sorted(consistent_classes),
                    "dropped_for_consistency": dropped}, indent=2),
        encoding="utf-8",
    )

    # Pass 2: save parquets
    for k in range(args.n_folds):
        fold_dir = args.out_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train = fold_data[k]["train"]
        test  = fold_data[k]["test"]
        method = fold_data[k]["method"]
        insuf  = fold_data[k]["insufficient"]

        bcol = args.behavior_column
        train = train.loc[train[bcol].astype(str).isin(consistent_classes)].reset_index(drop=True)
        test  = test.loc[ test[bcol].astype(str).isin(consistent_classes)].reset_index(drop=True)

        train, test, align = filter_test_behaviors_to_train(train, test, behavior_column=bcol)

        train.to_parquet(fold_dir / "train.parquet", engine="pyarrow",
                         compression="snappy", index=False)
        test.to_parquet( fold_dir / "test.parquet",  engine="pyarrow",
                         compression="snappy", index=False)

        # Verify anchors
        train_ids = set(train[args.subject_column].astype(str).unique())
        test_ids  = set(test[ args.subject_column].astype(str).unique())
        assert train_anchor in train_ids, f"fold_{k}: train anchor {train_anchor} missing from train!"
        assert test_anchor  in test_ids,  f"fold_{k}: test anchor {test_anchor} missing from test!"

        report = build_split_report(
            df, train, test,
            subject_column=args.subject_column,
            behavior_column=args.behavior_column,
            method={**method, "fold_index": k, "n_folds": args.n_folds,
                    "train_anchor": train_anchor, "test_anchor": test_anchor},
            test_label_alignment=align,
        )
        report["insufficient_train_behavior_filter"] = insuf
        report["consistent_classes"] = sorted(consistent_classes)
        report["behavior_label_mapping"] = label_meta
        (fold_dir / "split_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

        n_total = len(train) + len(test)
        play_train = int((train[bcol] == "Play").sum())
        play_test  = int((test[ bcol] == "Play").sum())
        print(
            f"\nfold_{k}: train={len(train):,} (Play={play_train:,})  "
            f"test={len(test):,} (Play={play_test:,})  "
            f"→ {fold_dir}"
        )


if __name__ == "__main__":
    main()
