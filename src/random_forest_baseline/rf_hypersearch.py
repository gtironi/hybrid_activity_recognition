"""RF baseline: hardcoded default vs hyperparameter search on the DL val split.

Reproduces the EXACT subject-level train/val split used by the deep-learning
pipeline (``_split_train_val_by_subject_gen_split``), tunes a RandomForest on that
val partition, and compares it against the current hardcoded default on the
held-out test partition.

Protocol (apples-to-apples with DL stage-1):
  * train-part = train minus val subjects; val used to *select*, not to fit.
  * ``*_tr``       models fit on the train-part only (matches DL: val for selection).
  * ``*_trainval`` models fit on train+val (matches the production RF baseline,
                   which today fits on the whole train parquet).
Selection metric defaults to balanced_accuracy (same as DL checkpoint selection).

Usage:
    PYTHONPATH=src python -m random_forest_baseline.rf_hypersearch \
        --train dataset/processed/AcTBeCalf/windowed_train.parquet \
        --test  dataset/processed/AcTBeCalf/windowed_test.parquet \
        --val_fraction 0.1 \
        --output_dir experiments/rf_hypersearch/AcTBeCalf
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from hybrid_activity_recognition.data.dataloader import _split_train_val_by_subject_gen_split
from hybrid_activity_recognition.training.evaluation_report import save_test_evaluation_artifacts
from hybrid_activity_recognition.training.metrics import classification_metrics_numpy
from random_forest_baseline.tsfel_baseline import _META_COLS, _feature_columns

# Current production baseline (random_forest_baseline.tsfel_baseline) hardcodes these.
DEFAULT_PARAMS = dict(
    n_estimators=200, max_depth=None, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced",
)

# Moderate default grid (32 configs). Override with --grid <json>.
DEFAULT_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 30],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.3],
    "class_weight": ["balanced", "balanced_subsample"],
}


def _subject_col(df: pd.DataFrame) -> str:
    for c in ("calf_id", "calfId", "subject", "subject_id"):
        if c in df.columns:
            return c
    raise SystemExit("No subject column (calf_id/calfId/subject) in train parquet.")


def _grid_configs(grid: dict, max_configs: int, seed: int) -> list[dict]:
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    if max_configs and 0 < max_configs < len(combos):
        random.Random(seed).shuffle(combos)
        combos = combos[:max_configs]
    return combos


def _fit_eval(params: dict, X_fit, y_fit, X_eval, y_eval, seed: int) -> tuple[RandomForestClassifier, dict, np.ndarray]:
    clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    clf.fit(X_fit, y_fit)
    y_pred = clf.predict(X_eval)
    return clf, classification_metrics_numpy(y_eval, y_pred), y_pred


def main():
    p = argparse.ArgumentParser(description="RF default-vs-tuned on the DL val split")
    p.add_argument("--train", required=True, help="Windowed train parquet")
    p.add_argument("--test", required=True, help="Windowed test parquet")
    p.add_argument("--val_fraction", type=float, default=0.2, help="Must match the DL run.")
    p.add_argument("--select_metric", default="balanced_accuracy",
                   choices=["balanced_accuracy", "f1_macro", "f1_weighted", "accuracy"])
    p.add_argument("--grid", type=str, default=None, help="JSON file overriding the search grid.")
    p.add_argument("--max_configs", type=int, default=0, help="Cap configs (0 = full grid).")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train)
    df_test = pd.read_parquet(args.test)
    df_train["label"] = df_train["label"].astype(str)
    df_test["label"] = df_test["label"].astype(str)

    feat_cols = _feature_columns(df_train)
    if not feat_cols:
        raise SystemExit("No TSFEL feature columns in the training parquet.")
    for c in feat_cols:
        if c not in df_test.columns:
            df_test[c] = 0.0

    subj = _subject_col(df_train)
    df_tr, df_val = _split_train_val_by_subject_gen_split(
        df_train, subject_column=subj, label_column="label", val_fraction=args.val_fraction,
    )
    val_ids = sorted(df_val[subj].astype(str).unique().tolist())
    print(f"subject_col={subj}  n_train_subj={df_train[subj].nunique()}  "
          f"val_subj={val_ids}  |tr|={len(df_tr)} |val|={len(df_val)} |test|={len(df_test)}")

    # Label space from full train (tr+val); drop test rows with unseen labels.
    le = LabelEncoder().fit(df_train["label"])
    known = set(le.classes_)
    n0 = len(df_test)
    df_test = df_test[df_test["label"].isin(known)].reset_index(drop=True)
    if len(df_test) < n0:
        print(f"Dropped {n0 - len(df_test)} test rows with unknown labels")

    def mat(df):
        X = np.nan_to_num(df[feat_cols].values.astype(np.float32), nan=0.0)
        return X, le.transform(df["label"])

    X_tr, y_tr = mat(df_tr)
    X_val, y_val = mat(df_val)
    X_trval, y_trval = mat(df_train)
    X_test, y_test = mat(df_test)

    # RF is scale-invariant, but mirror the baseline pipeline (fit scaler on the
    # fit-set, no leakage). Scaler fit on tr is reused for val scoring.
    sc_tr = StandardScaler().fit(X_tr)
    Xtr, Xval = sc_tr.transform(X_tr), sc_tr.transform(X_val)
    sc_trval = StandardScaler().fit(X_trval)
    Xtrval = sc_trval.transform(X_trval)
    Xtest_from_tr = sc_tr.transform(X_test)
    Xtest_from_trval = sc_trval.transform(X_test)

    # ---- hyperparameter search on val (fit on tr, score on val) ----
    grid = json.loads(Path(args.grid).read_text()) if args.grid else DEFAULT_GRID
    configs = _grid_configs(grid, args.max_configs, args.seed)
    print(f"Searching {len(configs)} configs, selecting by val {args.select_metric} ...")
    search = []
    best = None
    for i, params in enumerate(configs):
        _, vm, _ = _fit_eval(params, Xtr, y_tr, Xval, y_val, args.seed)
        search.append({"params": params, "val": vm})
        score = vm[args.select_metric]
        if best is None or score > best["score"]:
            best = {"score": score, "params": params, "val": vm}
        print(f"  [{i+1:02d}/{len(configs)}] val_{args.select_metric}={score:.4f}  {params}")
    print(f">>> best val_{args.select_metric}={best['score']:.4f}  params={best['params']}")

    # ---- final TEST evaluation of 4 models ----
    def evaluate(name, params, Xfit, yfit, Xtest):
        _, tm, y_pred = _fit_eval(params, Xfit, yfit, Xtest, y_test, args.seed)
        d = out / name
        d.mkdir(exist_ok=True)
        save_test_evaluation_artifacts(y_test, y_pred, le.classes_, d, stem="test")
        print(f"[{name:18}] acc={tm['accuracy']:.4f} balAcc={tm['balanced_accuracy']:.4f} "
              f"f1m={tm['f1_macro']:.4f} f1w={tm['f1_weighted']:.4f}")
        return tm

    results = {
        "default_tr":       evaluate("default_tr",       DEFAULT_PARAMS, Xtr,    y_tr,    Xtest_from_tr),
        "default_trainval": evaluate("default_trainval", DEFAULT_PARAMS, Xtrval, y_trval, Xtest_from_trval),
        "tuned_tr":         evaluate("tuned_tr",         best["params"], Xtr,    y_tr,    Xtest_from_tr),
        "tuned_trainval":   evaluate("tuned_trainval",   best["params"], Xtrval, y_trval, Xtest_from_trval),
    }

    summary = {
        "train_parquet": args.train, "test_parquet": args.test,
        "val_fraction": args.val_fraction, "subject_col": subj,
        "val_subjects": val_ids, "n_features": len(feat_cols),
        "select_metric": args.select_metric,
        "default_params": DEFAULT_PARAMS, "best_params": best["params"],
        "best_val_metrics": best["val"], "test_results": results,
        "search": search,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, default=str))

    # markdown table
    cols = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
    lines = [
        f"# RF default vs tuned — {Path(args.train).parent.name}",
        "",
        f"- val split reproduced from DL (`val_fraction={args.val_fraction}`, subjects: {', '.join(val_ids)})",
        f"- selection metric: **{args.select_metric}** · features: {len(feat_cols)}",
        f"- default params: `{DEFAULT_PARAMS}`",
        f"- **tuned params: `{best['params']}`** (val {args.select_metric}={best['score']:.4f})",
        "",
        "| Model | fit on | " + " | ".join(c.replace('_', ' ') for c in cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    fitset = {"default_tr": "train-part", "default_trainval": "train+val",
              "tuned_tr": "train-part", "tuned_trainval": "train+val"}
    for name, tm in results.items():
        row = f"| `{name}` | {fitset[name]} | " + " | ".join(f"{tm[c]*100:.2f}" for c in cols) + " |"
        lines.append(row)
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out/'comparison.json'} and {out/'comparison.md'}")


if __name__ == "__main__":
    main()
