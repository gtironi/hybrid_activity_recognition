#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _ensure_label_column(df: pd.DataFrame) -> str:
    for col in ("label", "activity", "target"):
        if col in df.columns:
            return col
    raise ValueError("No label column found. Expected one of: label, activity, target")


def main():
    parser = argparse.ArgumentParser(description="Generate top-k class filtered datasets.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k_values", default="5,7,10,13,15,18")
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    label_col = _ensure_label_column(train_df)

    if label_col not in test_df.columns:
        raise ValueError(f"Test parquet missing label column: {label_col}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = train_df[label_col].value_counts()
    ks = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    for k in ks:
        top_labels = list(counts.head(k).index)
        train_k = train_df[train_df[label_col].isin(top_labels)].reset_index(drop=True)
        test_k = test_df[test_df[label_col].isin(top_labels)].reset_index(drop=True)

        k_dir = out_dir / f"k_{k:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)
        train_path = k_dir / "train.parquet"
        test_path = k_dir / "test.parquet"
        meta_path = k_dir / "metadata.json"

        train_k.to_parquet(train_path, index=False)
        test_k.to_parquet(test_path, index=False)

        metadata = {
            "k": k,
            "label_column": label_col,
            "labels": top_labels,
            "train_rows": int(len(train_k)),
            "test_rows": int(len(test_k)),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved top-k variants to: {out_dir}")


if __name__ == "__main__":
    main()
