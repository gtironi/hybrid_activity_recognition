#!/usr/bin/env python3
"""
Generate K non-overlapping subject-level folds for cross-validation.

For each fold k in [0, K-1]:
  - fold_k's subjects become the test set
  - the remaining subjects become the train set
  - same canonical label mapping + rare-behavior filter as dataset_processing.py
  - saves train.parquet, test.parquet, split_report.json into {out_dir}/fold_{k}/

Also writes {out_dir}/fold_assignments.json mapping fold name -> test subject IDs.

Reuses the existing functions in dataset_processing.py so behaviour stays
identical to the single-split pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import genSplit
from dataset_processing import (
    DEFAULT_BEHAVIOR_COLUMN,
    DEFAULT_MIN_TRAIN_PROPORTION_PER_BEHAVIOR,
    _subject_behavior_wide,
    apply_canonical_behavior_labels,
    build_split_report,
    filter_behaviors_below_min_train_proportion,
    filter_test_behaviors_to_train,
    split_subject_list,
)


def main() -> None:
    p = argparse.ArgumentParser(description="K-fold subject-level splits → fold parquets")
    p.add_argument("--csv", type=Path, default=Path("dataset/AcTBeCalf.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("dataset/processed/kfold"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--subject-column", default="calfId")
    p.add_argument("--behavior-column", default=DEFAULT_BEHAVIOR_COLUMN)
    p.add_argument(
        "--min-train-proportion-per-behavior",
        type=float,
        default=DEFAULT_MIN_TRAIN_PROPORTION_PER_BEHAVIOR,
    )
    p.add_argument(
        "--label-mapping",
        choices=("calf", "none"),
        default="calf",
        help="calf = apply BEHAVIOUR_LABEL_MAP; none = passthrough.",
    )
    p.add_argument(
        "--reuse-assignments",
        type=Path,
        default=None,
        help="Reuse test subjects from an existing fold_assignments.json instead of "
        "recomputing the partition (keeps folds identical across experiments).",
    )
    p.add_argument(
        "--skip-rare-filter",
        action="store_true",
        help="Skip the rare-behavior filter and consistent-class intersection. Saves "
        "subject-disjoint canonical splits with the original behavior column intact "
        "(for pipelines that remap labels downstream, e.g. paper 6-class).",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.csv} ...")
    df = pd.read_csv(args.csv)

    if args.label_mapping == "calf":
        label_meta = apply_canonical_behavior_labels(df, args.behavior_column)
    else:
        df[args.behavior_column] = df[args.behavior_column].astype(str)
        label_meta = {
            "mode": "none",
            "raw_classes": sorted(df[args.behavior_column].unique().tolist(), key=str),
        }

    wide = _subject_behavior_wide(df, args.subject_column, args.behavior_column)
    subjects = sorted(wide["subject_id"].unique().tolist(), key=str)
    n_sub = len(subjects)
    if n_sub < args.n_folds:
        raise SystemExit(f"Not enough subjects ({n_sub}) for {args.n_folds} folds.")

    if args.reuse_assignments is not None:
        if not args.reuse_assignments.is_file():
            raise SystemExit(f"--reuse-assignments not found: {args.reuse_assignments}")
        src = json.loads(args.reuse_assignments.read_text(encoding="utf-8"))
        per_fold = src["test_subjects_per_fold"]
        n_folds_src = len(per_fold)
        if n_folds_src != args.n_folds:
            raise SystemExit(
                f"--reuse-assignments has {n_folds_src} folds but --n-folds={args.n_folds}"
            )
        # Coerce reused subject IDs back to the dtype of the subject column.
        subj_dtype = type(subjects[0])
        folds = [
            [subj_dtype(s) for s in per_fold[f"fold_{k}"]] for k in range(args.n_folds)
        ]
        print(f"Reusing fold assignments from {args.reuse_assignments}")
    else:
        folds = genSplit.partition_subjects_into_folds(subjects, args.n_folds, wide)

    assignments = {f"fold_{k}": sorted(folds[k], key=str) for k in range(args.n_folds)}
    assignments_path = args.out_dir / "fold_assignments.json"
    assignments_path.write_text(
        json.dumps(
            {
                "n_folds": args.n_folds,
                "n_subjects_total": n_sub,
                "subject_column": args.subject_column,
                "behavior_column": args.behavior_column,
                "test_subjects_per_fold": assignments,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nFold assignments written: {assignments_path}")

    print("\nFold summary:")
    print(f"{'Fold':<8} {'n_test':>7} {'test subjects'}")
    print("-" * 80)
    for k in range(args.n_folds):
        ids = assignments[f"fold_{k}"]
        print(f"fold_{k:<3} {len(ids):>7}  {ids}")

    # --- Simple mode: subject-disjoint canonical splits, no filtering. The
    # original behavior column is preserved so downstream pipelines can remap
    # labels themselves (e.g. paper 6-class). ---
    if args.skip_rare_filter:
        for k in range(args.n_folds):
            fold_dir = args.out_dir / f"fold_{k}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train, test, method = split_subject_list(
                df,
                subject_column=args.subject_column,
                test_subjects=[str(s) for s in folds[k]],
            )
            train.to_parquet(
                fold_dir / "train.parquet", engine="pyarrow", compression="snappy", index=False
            )
            test.to_parquet(
                fold_dir / "test.parquet", engine="pyarrow", compression="snappy", index=False
            )
            report = build_split_report(
                df,
                train,
                test,
                subject_column=args.subject_column,
                behavior_column=args.behavior_column,
                method={**method, "fold_index": k, "n_folds": args.n_folds},
                test_label_alignment=None,
            )
            report["behavior_label_mapping"] = label_meta
            (fold_dir / "split_report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            n_total = len(train) + len(test)
            print(
                f"\nfold_{k}: train={len(train):,} ({100*len(train)/n_total:.1f}%)  "
                f"test={len(test):,} ({100*len(test)/n_total:.1f}%)  "
                f"{train[args.behavior_column].nunique()} raw classes  → {fold_dir}"
            )
        return

    # --- Pass 1: split each fold and apply the per-fold rare-behavior filter
    # to discover which classes survive. The intersection across folds is the
    # consistent class set used for every fold, so all folds solve the same
    # N-class problem (comparable mean ± std). ---
    fold_data: list[dict] = []
    surviving_sets: list[set] = []
    for k in range(args.n_folds):
        test_subjects = folds[k]
        train, test, method = split_subject_list(
            df,
            subject_column=args.subject_column,
            test_subjects=[str(s) for s in test_subjects],
        )
        train, test, insufficient_meta = filter_behaviors_below_min_train_proportion(
            train,
            test,
            behavior_column=args.behavior_column,
            min_train_proportion=args.min_train_proportion_per_behavior,
        )
        surviving = set(train[args.behavior_column].astype(str).unique())
        surviving_sets.append(surviving)
        fold_data.append(
            {"train": train, "test": test, "method": method, "insufficient": insufficient_meta}
        )

    consistent_classes = set.intersection(*surviving_sets)
    print(f"\nConsistent class set across all folds ({len(consistent_classes)}):")
    for c in sorted(consistent_classes, key=str):
        print(f"  - {c}")
    dropped_for_consistency = sorted(set.union(*surviving_sets) - consistent_classes, key=str)
    if dropped_for_consistency:
        print(f"Dropped to keep folds consistent: {dropped_for_consistency}")

    (args.out_dir / "consistent_classes.json").write_text(
        json.dumps(
            {
                "consistent_classes": sorted(consistent_classes, key=str),
                "dropped_for_consistency": dropped_for_consistency,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # --- Pass 2: restrict every fold to the consistent class set and save. ---
    for k in range(args.n_folds):
        fold_dir = args.out_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train = fold_data[k]["train"]
        test = fold_data[k]["test"]
        method = fold_data[k]["method"]
        insufficient_meta = fold_data[k]["insufficient"]

        beh_train = train[args.behavior_column].astype(str)
        beh_test = test[args.behavior_column].astype(str)
        train = train.loc[beh_train.isin(consistent_classes)].reset_index(drop=True)
        test = test.loc[beh_test.isin(consistent_classes)].reset_index(drop=True)

        train, test, alignment_meta = filter_test_behaviors_to_train(
            train, test, behavior_column=args.behavior_column
        )

        train.to_parquet(
            fold_dir / "train.parquet", engine="pyarrow", compression="snappy", index=False
        )
        test.to_parquet(
            fold_dir / "test.parquet", engine="pyarrow", compression="snappy", index=False
        )

        report = build_split_report(
            df,
            train,
            test,
            subject_column=args.subject_column,
            behavior_column=args.behavior_column,
            method={**method, "fold_index": k, "n_folds": args.n_folds},
            test_label_alignment=alignment_meta,
        )
        report["insufficient_train_behavior_filter"] = insufficient_meta
        report["consistent_classes"] = sorted(consistent_classes, key=str)
        report["behavior_label_mapping"] = label_meta
        (fold_dir / "split_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

        n_total = len(train) + len(test)
        print(
            f"\nfold_{k}: train={len(train):,} ({100*len(train)/n_total:.1f}%)  "
            f"test={len(test):,} ({100*len(test)/n_total:.1f}%)  "
            f"{train[args.behavior_column].nunique()} classes  → {fold_dir}"
        )


if __name__ == "__main__":
    main()
