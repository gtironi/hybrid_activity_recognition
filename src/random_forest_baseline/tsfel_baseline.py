"""TSFEL-only baseline: SelectKBest + RandomForest.

Standalone script — no PyTorch dependency.  Reads the same windowed parquets
used by the deep learning experiments, extracts only the TSFEL feature columns,
selects the top-k with SelectKBest (f_classif), and fits a RandomForest.

Usage:
    PYTHONPATH=src python -m random_forest_baseline.tsfel_baseline \
        --train dataset/processed/AcTBeCalf/windowed_train.parquet \
        --test  dataset/processed/AcTBeCalf/windowed_test.parquet \
        --output_dir experiments/tsfel_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Same meta columns used by the DL pipeline
_META_COLS = frozenset(
    {"dateTime", "calfId", "calf_id", "segId", "segment_id", "subject",
     "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
     "mag_x", "mag_y", "mag_z", "label", "datetime", "window_start"}
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def main():
    p = argparse.ArgumentParser(description="TSFEL baseline (SelectKBest + RandomForest)")
    p.add_argument("--train", required=True, help="Windowed parquet for training")
    p.add_argument("--test", required=True, help="Windowed parquet for testing")
    p.add_argument("--n_estimators", type=int, default=200, help="RandomForest n_estimators")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="experiments/tsfel_baseline")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    df_train = pd.read_parquet(args.train)
    df_test = pd.read_parquet(args.test)

    feat_cols = _feature_columns(df_train)
    if not feat_cols:
        raise SystemExit("No TSFEL feature columns found in the training parquet.")

    # Align columns
    for c in feat_cols:
        if c not in df_test.columns:
            df_test[c] = 0.0

    # Encode labels
    le = LabelEncoder()
    le.fit(df_train["label"])
    known = set(le.classes_)

    n_before = len(df_test)
    df_test = df_test[df_test["label"].isin(known)].reset_index(drop=True)
    if len(df_test) < n_before:
        print(f"Dropped {n_before - len(df_test)} test rows with unknown labels")

    y_train = le.transform(df_train["label"])
    y_test = le.transform(df_test["label"])

    X_train = df_train[feat_cols].values.astype(np.float32)
    X_test = df_test[feat_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Using all {len(feat_cols)} TSFEL features")

    # Train
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"accuracy={acc:.4f}")
    print(f"f1_macro={f1_macro:.4f}")
    print(f"f1_weighted={f1_weighted:.4f}")

    # Save results
    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "n_estimators": args.n_estimators,
        "n_features": len(feat_cols),
        "class_names": le.classes_.tolist(),
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
