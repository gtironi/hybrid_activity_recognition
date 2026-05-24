"""Late fusion of deep-model probabilities with TSFEL Random Forest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

_META_COLS = frozenset(
    {
        "dateTime",
        "calfId",
        "calf_id",
        "segId",
        "segment_id",
        "subject",
        "acc_x",
        "acc_y",
        "acc_z",
        "label",
    }
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def fit_tsfel_random_forest_from_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_estimators: int = 200,
    seed: int = 42,
) -> dict:
    """Fit RF on the same train split as the deep model (no val leakage)."""
    x = np.nan_to_num(features.astype(np.float32), nan=0.0)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(x, labels)
    return {
        "classifier": clf,
        "scaler": scaler,
        "label_encoder": None,
        "feat_cols": None,
    }


def align_rf_proba_to_class_indices(
    proba_rf: np.ndarray,
    rf_class_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Map RF ``predict_proba`` columns (forest class order) to 0..num_classes-1."""
    out = np.zeros((proba_rf.shape[0], num_classes), dtype=np.float64)
    for col_j, cls_idx in enumerate(rf_class_indices):
        idx = int(cls_idx)
        if 0 <= idx < num_classes:
            out[:, idx] = proba_rf[:, col_j]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


def fit_tsfel_random_forest(
    train_parquet: str | Path,
    *,
    n_estimators: int = 200,
    seed: int = 42,
) -> dict:
    """Train RF on all TSFEL columns; returns artifact dict (in-memory)."""
    df = pd.read_parquet(train_parquet)
    feat_cols = _feature_columns(df)
    if not feat_cols:
        raise ValueError("No TSFEL feature columns found in training parquet.")

    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str))
    x = np.nan_to_num(df[feat_cols].values.astype(np.float32), nan=0.0)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(x, y)
    return {
        "classifier": clf,
        "scaler": scaler,
        "label_encoder": le,
        "feat_cols": feat_cols,
    }


def save_rf_artifacts(artifacts: dict, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts["classifier"], out / "rf_model.joblib")
    joblib.dump(artifacts["scaler"], out / "rf_scaler.joblib")
    if artifacts.get("label_encoder") is not None:
        joblib.dump(artifacts["label_encoder"], out / "rf_label_encoder.joblib")
    if artifacts.get("feat_cols") is not None:
        with open(out / "rf_feat_cols.json", "w", encoding="utf-8") as f:
            json.dump(artifacts["feat_cols"], f)
    return out


def load_rf_artifacts(artifact_dir: str | Path) -> dict:
    root = Path(artifact_dir)
    art = {
        "classifier": joblib.load(root / "rf_model.joblib"),
        "scaler": joblib.load(root / "rf_scaler.joblib"),
        "label_encoder": None,
        "feat_cols": None,
    }
    le_path = root / "rf_label_encoder.joblib"
    if le_path.is_file():
        art["label_encoder"] = joblib.load(le_path)
    feat_path = root / "rf_feat_cols.json"
    if feat_path.is_file():
        with open(feat_path, encoding="utf-8") as f:
            art["feat_cols"] = json.load(f)
    return art


def rf_proba_from_parquet(artifacts: dict, parquet_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true indices, proba) aligned to sklearn classes 0..C-1."""
    df = pd.read_parquet(parquet_path)
    feat_cols = artifacts["feat_cols"]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    le: LabelEncoder = artifacts["label_encoder"]
    known = set(le.classes_)
    mask = df["label"].astype(str).isin(known)
    df = df.loc[mask].reset_index(drop=True)
    y = le.transform(df["label"].astype(str))
    x = np.nan_to_num(df[feat_cols].values.astype(np.float32), nan=0.0)
    x = artifacts["scaler"].transform(x)
    proba = artifacts["classifier"].predict_proba(x)
    return y, proba


def rf_proba_from_features(artifacts: dict, features: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(features.astype(np.float32), nan=0.0)
    x = artifacts["scaler"].transform(x)
    return artifacts["classifier"].predict_proba(x)


def align_proba_to_labels(
    proba: np.ndarray,
    source_classes: np.ndarray,
    target_num_classes: int,
) -> np.ndarray:
    """Map proba columns from ``source_classes`` indices to ``0..target_num_classes-1``."""
    out = np.zeros((proba.shape[0], target_num_classes), dtype=np.float64)
    for j, cls_idx in enumerate(source_classes):
        if 0 <= int(cls_idx) < target_num_classes:
            out[:, int(cls_idx)] = proba[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


def soft_vote_proba(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    weight_a: float = 0.5,
) -> np.ndarray:
    w = float(weight_a)
    combined = w * proba_a + (1.0 - w) * proba_b
    combined = np.clip(combined, 1e-8, None)
    return combined / combined.sum(axis=1, keepdims=True)


def fit_stacking_logistic(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 500,
) -> LogisticRegression:
    x = np.hstack([proba_a, proba_b])
    meta = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    meta.fit(x, y)
    return meta


def stacking_predict(
    meta: LogisticRegression,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
) -> np.ndarray:
    x = np.hstack([proba_a, proba_b])
    return meta.predict_proba(x)


def ensemble_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def combine_and_predict(
    proba_deep: np.ndarray,
    proba_rf: np.ndarray,
    y_true: np.ndarray,
    *,
    method: Literal["soft_vote", "stacking"] = "soft_vote",
    weight_deep: float = 0.5,
    meta: LogisticRegression | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "soft_vote":
        proba = soft_vote_proba(proba_deep, proba_rf, weight_a=weight_deep)
    elif method == "stacking":
        if meta is None:
            raise ValueError("stacking requires a fitted meta learner")
        proba = stacking_predict(meta, proba_deep, proba_rf)
    else:
        raise ValueError(f"Unknown ensemble method: {method!r}")
    return proba, proba.argmax(axis=1)
