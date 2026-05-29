#!/usr/bin/env python3
"""
Parquet bruto (série longa) -> Parquet janelado + features do paper Dissanayake et al. (2025).

Replica HC / Catch22 / miniROCKET seguindo `calfBehaviourEval/`:
- 8 séries derivadas por amostra (sem filtro Butterworth, como no notebook do paper):
    accX, accY, accZ, Amag, ODBA, VeDBA, pitch, roll
- Mesmas regras de janelamento que `prepare_windowed_parquet.py` (purity_threshold no rótulo dominante).

Saída: `dateTime, calf_id, acc_x (list), acc_y (list), acc_z (list), label, [feature cols...]`.

Modos:
  --features hc        -> 11 estatísticas x 8 séries = 88 colunas (`hc_<serie>_<stat>`)
  --features catch22   -> 24 features (catch24) x 8 séries = 192 colunas (`c22_<serie>_<NN>`)
  --features rocket    -> MiniRocket(num_kernels=10000) -> 9996 colunas (`rocket_<i>`)
                          treino: fit + transform + joblib.dump em --rocket-manifest-out
                          teste:  joblib.load de --rocket-manifest-in + transform

Um ficheiro de entrada por execução.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import the paper's own feature functions to stay 1:1 with their code.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CALF_LIB = _REPO_ROOT / "calfBehaviourEval" / "libraries"
if str(_CALF_LIB) not in sys.path:
    sys.path.insert(0, str(_CALF_LIB))

from functions import (  # noqa: E402
    calculate_magnitude,
    calculate_ODBA,
    calculate_VeDBA,
    calculate_pitch,
    calculate_roll,
    contains_non_numeric,
)
from hand_crafted_features import return_HC_features  # noqa: E402

# 11 HC stats, in the order `return_HC_features` appends them.
HC_STAT_NAMES = (
    "mean", "median", "min", "max", "std",
    "q1", "q3", "spectral_entropy", "motion_variation",
    "kurtosis", "skewness",
)
# Order matches calfBehaviourEval/Feature datasets derivation.ipynb (`keys` constant).
SERIES_KEYS = ("accX", "accY", "accZ", "Amag", "ODBA", "VeDBA", "pitch", "roll")

# Silence noisy moment / divide warnings from scipy on flat windows.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Derived signals (per sample, from raw X/Y/Z — no filtering, paper's recipe).
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame, *, acc_x: str, acc_y: str, acc_z: str) -> pd.DataFrame:
    """Add Amag, ODBA, VeDBA, pitch, roll columns to ``df`` in vectorized form."""
    x = df[acc_x].to_numpy(dtype=np.float64)
    y = df[acc_y].to_numpy(dtype=np.float64)
    z = df[acc_z].to_numpy(dtype=np.float64)

    df["Amag"] = calculate_magnitude(x, y, z)
    df["ODBA"] = calculate_ODBA(x, y, z)
    df["VeDBA"] = calculate_VeDBA(x, y, z)  # same formula as Amag when called with raw values

    # calculate_pitch/roll use math.atan2 (scalar); vectorize via np.arctan2 to match.
    df["pitch"] = 180.0 * np.arctan2(z, np.sqrt(y * y + x * x)) / math.pi
    df["roll"] = 180.0 * np.arctan2(y, np.sqrt(x * x + z * z)) / math.pi
    return df


# ---------------------------------------------------------------------------
# Windowing — same rules as scripts/prepare_windowed_parquet.py but over 8 cols.
# ---------------------------------------------------------------------------

def create_windowed_records(
    df: pd.DataFrame,
    *,
    window_size: int,
    overlap: float,
    purity_threshold: float,
    time_column: str,
    group_by: list[str],
    label_column: str,
    acc_x: str,
    acc_y: str,
    acc_z: str,
) -> list[dict]:
    """Slide a fixed-length window over each segment; return one dict per kept window.

    Each dict has ``dateTime, calf_id, label`` plus an array per series key in SERIES_KEYS.
    """
    stride = int(window_size * (1 - overlap))
    if stride <= 0:
        raise SystemExit(f"Stride <= 0 (window_size={window_size}, overlap={overlap}).")

    cols = [*group_by, time_column, acc_x, acc_y, acc_z, label_column]
    series_arr_cols = [acc_x, acc_y, acc_z, "Amag", "ODBA", "VeDBA", "pitch", "roll"]

    # Use only what we need to keep memory low.
    if any(c not in df.columns for c in cols):
        missing = [c for c in cols if c not in df.columns]
        raise SystemExit(f"Colunas em falta no input: {missing}")
    df = df[cols + ["Amag", "ODBA", "VeDBA", "pitch", "roll"]].copy()
    df[label_column] = df[label_column].astype(str)

    records: list[dict] = []
    for key, group in df.groupby(group_by, sort=False):
        if len(group_by) == 1:
            key = (key,)
        calf_id = key[0]
        n_samples = len(group)
        if n_samples < window_size:
            continue

        series = {sk_out: group[sc_in].to_numpy() for sk_out, sc_in in zip(SERIES_KEYS, series_arr_cols)}
        timestamps = group[time_column].to_numpy()
        labels = group[label_column].to_numpy()

        for start_idx in range(0, n_samples - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_labels = labels[start_idx:end_idx]
            counts = Counter(window_labels)
            most_common_label, count = counts.most_common(1)[0]
            if (count / window_size) < purity_threshold:
                continue
            rec = {
                "dateTime": timestamps[start_idx],
                "calf_id": calf_id,
                "label": most_common_label,
            }
            for sk in SERIES_KEYS:
                rec[sk] = series[sk][start_idx:end_idx]
            records.append(rec)

    print(f"Janelas geradas: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Feature extractors.
# ---------------------------------------------------------------------------

def compute_hc_features(records: list[dict]) -> tuple[np.ndarray, list[str], list[int]]:
    """Returns (X, feat_names, kept_indices). Drops windows with any incomplete HC series."""
    feat_names = [f"hc_{sk}_{stat}" for sk in SERIES_KEYS for stat in HC_STAT_NAMES]
    n_feat = len(feat_names)
    n_per_series = len(HC_STAT_NAMES)

    rows: list[np.ndarray] = []
    kept_idx: list[int] = []
    for i, rec in enumerate(records):
        flat = np.empty(n_feat, dtype=np.float64)
        ok = True
        for j, sk in enumerate(SERIES_KEYS):
            feats = return_HC_features(np.asarray(rec[sk], dtype=np.float64))
            if len(feats) != n_per_series:
                ok = False
                break
            flat[j * n_per_series:(j + 1) * n_per_series] = feats
        if ok and not contains_non_numeric(flat.tolist()):
            rows.append(flat)
            kept_idx.append(i)

    X = np.stack(rows, axis=0) if rows else np.empty((0, n_feat), dtype=np.float64)
    dropped = len(records) - len(kept_idx)
    if dropped:
        print(f"  HC: dropped {dropped} windows with incomplete features")
    return X, feat_names, kept_idx


def compute_catch22_features(records: list[dict]) -> tuple[np.ndarray, list[str], list[int]]:
    """Returns (X, feat_names, kept_indices) for catch24 (22 + mean + std)."""
    from pycatch22 import catch22_all
    # One probe call to read the official feature names from pycatch22.
    probe = catch22_all([0.0] * 8, catch24=True)
    n_per_series = len(probe["values"])  # 24
    raw_names = probe["names"]  # length 24
    feat_names = [f"c22_{sk}_{nm}" for sk in SERIES_KEYS for nm in raw_names]
    n_feat = len(feat_names)

    rows: list[np.ndarray] = []
    kept_idx: list[int] = []
    for i, rec in enumerate(records):
        flat = np.empty(n_feat, dtype=np.float64)
        ok = True
        for j, sk in enumerate(SERIES_KEYS):
            try:
                vals = catch22_all(list(rec[sk]), catch24=True)["values"]
            except Exception:
                ok = False
                break
            if len(vals) != n_per_series:
                ok = False
                break
            flat[j * n_per_series:(j + 1) * n_per_series] = vals
        if ok and not contains_non_numeric(flat.tolist()):
            rows.append(flat)
            kept_idx.append(i)

    X = np.stack(rows, axis=0) if rows else np.empty((0, n_feat), dtype=np.float64)
    dropped = len(records) - len(kept_idx)
    if dropped:
        print(f"  Catch22: dropped {dropped} windows with non-numeric features")
    return X, feat_names, kept_idx


def _stack_for_rocket(records: list[dict]) -> np.ndarray:
    """Stack records into (n_windows, 8, window_size) float32 for MiniRocket."""
    n = len(records)
    w = len(records[0][SERIES_KEYS[0]])
    X = np.empty((n, len(SERIES_KEYS), w), dtype=np.float32)
    for i, rec in enumerate(records):
        for j, sk in enumerate(SERIES_KEYS):
            X[i, j, :] = rec[sk]
    return X


def compute_rocket_features(
    records: list[dict],
    *,
    n_kernels: int,
    random_state: int,
    manifest_in: Path | None,
    manifest_out: Path | None,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Fit (train) or load (test) MiniRocket and transform all windows."""
    from aeon.transformations.collection.convolution_based import MiniRocket

    if not records:
        return np.empty((0, 0), dtype=np.float32), [], []

    X = _stack_for_rocket(records)
    if manifest_in is not None:
        print(f"  ROCKET: loading fitted model from {manifest_in}")
        rocket = joblib.load(manifest_in)
    else:
        print(f"  ROCKET: fitting MiniRocket(n_kernels={n_kernels}, random_state={random_state})")
        rocket = MiniRocket(n_kernels=n_kernels, random_state=random_state)
        rocket.fit(X)
        if manifest_out is not None:
            manifest_out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(rocket, manifest_out)
            print(f"  ROCKET: saved fitted model to {manifest_out.resolve()}")
        else:
            print("  AVISO: sem --rocket-manifest-out o conjunto de teste não poderá reaproveitar este modelo.")

    Xt = rocket.transform(X)
    if hasattr(Xt, "to_numpy"):
        Xt = Xt.to_numpy()
    Xt = np.asarray(Xt, dtype=np.float32)
    feat_names = [f"rocket_{i}" for i in range(Xt.shape[1])]
    return Xt, feat_names, list(range(len(records)))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def _build_output_df(records: list[dict], kept_idx: list[int], X: np.ndarray, feat_names: list[str]) -> pd.DataFrame:
    """Assemble final dataframe: meta cols + raw acc lists + feature columns."""
    meta = []
    for i in kept_idx:
        rec = records[i]
        meta.append({
            "dateTime": rec["dateTime"],
            "calf_id": rec["calf_id"],
            "acc_x": rec["accX"].tolist(),
            "acc_y": rec["accY"].tolist(),
            "acc_z": rec["accZ"].tolist(),
            "label": rec["label"],
        })
    df_meta = pd.DataFrame(meta)
    if X.size == 0:
        return df_meta
    df_feat = pd.DataFrame(X, columns=feat_names)
    return pd.concat([df_meta.reset_index(drop=True), df_feat.reset_index(drop=True)], axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description="Parquet bruto -> janelado + features do paper (HC/Catch22/ROCKET)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--features", choices=("hc", "catch22", "rocket"), required=True)
    p.add_argument("--rocket-manifest-out", type=Path, default=None,
                   help="Caminho onde gravar o MiniRocket fitted (modo treino).")
    p.add_argument("--rocket-manifest-in", type=Path, default=None,
                   help="Caminho do MiniRocket fitted a usar (modo teste).")
    p.add_argument("--rocket-n-kernels", type=int, default=10000)
    p.add_argument("--rocket-random-state", type=int, default=42)
    p.add_argument("--window-size", type=int, default=125, help="5 s @ 25 Hz = 125 amostras (default do paper).")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--purity-threshold", type=float, default=0.9)
    p.add_argument("--fs", type=int, default=25, help="Sampling rate (informacional).")
    p.add_argument("--group-by", nargs="+", default=["calfId", "segId"])
    p.add_argument("--time-column", default="dateTime")
    p.add_argument("--label-column", default="behaviour")
    p.add_argument("--acc-x", dest="acc_x", default="accX")
    p.add_argument("--acc-y", dest="acc_y", default="accY")
    p.add_argument("--acc-z", dest="acc_z", default="accZ")
    p.add_argument("--remap-labels", type=Path, default=None,
                   help="JSON file {raw_label: new_label}. Labels not in map become 'Other'.")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Entrada não encontrada: {args.input}")
    if args.features == "rocket" and args.rocket_manifest_in is not None and not args.rocket_manifest_in.is_file():
        raise SystemExit(f"Manifest ROCKET não encontrado: {args.rocket_manifest_in}")

    print(f"Lendo {args.input}...")
    cols_needed = list({*args.group_by, args.time_column, args.acc_x, args.acc_y, args.acc_z, args.label_column})
    df = pd.read_parquet(args.input, columns=cols_needed)
    print(f"  linhas: {len(df):,}")

    if args.remap_labels is not None:
        import json
        mapping = json.loads(args.remap_labels.read_text())
        df[args.label_column] = df[args.label_column].map(lambda x: mapping.get(x, "Other"))
        print(f"Labels remapped via {args.remap_labels}: {df[args.label_column].value_counts().to_dict()}")

    print("Derivando 8 séries (sem filtro, igual ao notebook do paper)...")
    df = add_derived_columns(df, acc_x=args.acc_x, acc_y=args.acc_y, acc_z=args.acc_z)

    print(f"Janelando (window={args.window_size}, overlap={args.overlap}, purity>={args.purity_threshold})...")
    records = create_windowed_records(
        df,
        window_size=args.window_size,
        overlap=args.overlap,
        purity_threshold=args.purity_threshold,
        time_column=args.time_column,
        group_by=args.group_by,
        label_column=args.label_column,
        acc_x=args.acc_x,
        acc_y=args.acc_y,
        acc_z=args.acc_z,
    )
    if not records:
        raise SystemExit("Nenhuma janela gerada; verifique dados e hiperparâmetros.")

    print(f"Calculando features ({args.features})...")
    if args.features == "hc":
        X, feat_names, kept = compute_hc_features(records)
    elif args.features == "catch22":
        X, feat_names, kept = compute_catch22_features(records)
    else:  # rocket
        X, feat_names, kept = compute_rocket_features(
            records,
            n_kernels=args.rocket_n_kernels,
            random_state=args.rocket_random_state,
            manifest_in=args.rocket_manifest_in,
            manifest_out=args.rocket_manifest_out,
        )

    if not kept:
        raise SystemExit("Nenhuma janela sobreviveu ao filtro de features.")

    df_out = _build_output_df(records, kept, X, feat_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, engine="pyarrow", compression="snappy", index=False)
    print(f"Salvo: {args.output.resolve()} | shape={df_out.shape} | feature_cols={len(feat_names)}")


if __name__ == "__main__":
    main()
