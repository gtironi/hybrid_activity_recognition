"""Canonical dataset layout: meta.json + train/val/test CSVs for PatchTST classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """hybrid_activity_recognition/ (parent of hugging/)."""
    return package_root().parent.parent


def feature_column_names(context_length: int, num_channels: int) -> list[str]:
    n = context_length * num_channels
    return [f"x_{i:03d}" for i in range(n)]


def flatten_time_major(window: np.ndarray) -> np.ndarray:
    """(T, C) -> (T*C,) in time-major order: t0c0..t0c_{C-1}, t1c0,..."""
    if window.ndim != 2:
        raise ValueError("window must be 2-D (T, C)")
    t, c = window.shape
    return window.reshape(t * c).astype(np.float64, copy=False)


def windows_to_feature_matrix(windows: np.ndarray) -> np.ndarray:
    """(N, T, C) -> (N, T*C) time-major flattened rows."""
    if windows.ndim != 3:
        raise ValueError("windows must be (N, T, C)")
    n, t, c = windows.shape
    return windows.reshape(n, t * c).astype(np.float64, copy=False)


def save_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    with open(data_dir / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def write_split_csv(
    out_path: Path,
    labels: np.ndarray,
    features: np.ndarray,
    *,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> None:
    """Write one split CSV: label, optional extras, x_000 ..."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, d = features.shape
    cols: dict[str, Any] = {"label": labels.astype(np.int64)}
    if extra_columns:
        for k, v in extra_columns.items():
            cols[k] = v
    for i in range(d):
        cols[f"x_{i:03d}"] = features[:, i]
    pd.DataFrame(cols).to_csv(out_path, index=False)


def load_csv_tensors(
    csv_path: Path,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, T*C) float32, y (N,) int64 using meta feature columns only."""
    df = pd.read_csv(csv_path)
    y = df["label"].to_numpy(dtype=np.int64)
    feat_cols = feature_column_names(meta["context_length"], meta["num_channels"])
    x = df[feat_cols].to_numpy(dtype=np.float32)
    return x, y


def tensors_to_past_values(x_flat: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """(N, T*C) -> (N, T, C) for Hugging Face past_values."""
    t = meta["context_length"]
    c = meta["num_channels"]
    return x_flat.reshape(-1, t, c)


def print_train_hint(data_dir: Path, preset: str | None = None) -> None:
    data_dir = Path(data_dir).resolve()
    preset_arg = f" --preset {preset}" if preset else f" --data_dir {data_dir}"
    print("\nNext step (from repo root):")
    print(f"  python -m hugging.patchtst.train_classification_debug{preset_arg}\n")
