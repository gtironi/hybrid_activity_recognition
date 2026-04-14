#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_array(cell):
    arr = np.asarray(cell, dtype=np.float32)
    return arr


def _per_window_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("acc_x", "acc_y", "acc_z"):
        normalized = []
        for cell in out[col].values:
            arr = _to_array(cell)
            m = float(arr.mean())
            s = float(arr.std())
            normalized.append(((arr - m) / (s + 1e-6)).astype(np.float32))
        out[col] = normalized
    return out


def _save(df_train: pd.DataFrame, df_test: pd.DataFrame, out_dir: Path, variant: str):
    d = out_dir / variant
    d.mkdir(parents=True, exist_ok=True)
    train_path = d / "train.parquet"
    test_path = d / "test.parquet"
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate normalization variants for ablations.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Variant 1: original input as-is.
    _save(train_df, test_df, out_dir, "global_pipeline")

    # Variant 2: per-window z-score before pipeline normalization.
    train_pw = _per_window_zscore(train_df)
    test_pw = _per_window_zscore(test_df)
    _save(train_pw, test_pw, out_dir, "per_window_zscore")

    print(f"Saved normalization variants to: {out_dir}")


if __name__ == "__main__":
    main()
