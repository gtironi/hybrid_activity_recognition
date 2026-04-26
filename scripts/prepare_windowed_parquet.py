#!/usr/bin/env python3
"""
Parquet bruto (série longa) → Parquet janelado + TSFEL, alinhado ao legado window_creator_labeled.

Modo discover (sem --feature-manifest-in): amostra + RF escolhe top-N colunas TSFEL; grava manifest JSON.
Modo apply (--feature-manifest-in): só extrai as colunas do manifest (para teste sem vazamento).
Skewness/Kurtosis (domínios statistical e spectral) não entram na extração TSFEL.

Um ficheiro de entrada por execução.

Exemplos (AcTBeCalf em dataset/processed/AcTBeCalf/):

  python scripts/prepare_windowed_parquet.py \\
    --input dataset/processed/AcTBeCalf/train.parquet \\
    --output dataset/processed/AcTBeCalf/windowed_train.parquet \\
    --feature-manifest-out dataset/processed/AcTBeCalf/tsfel_feature_manifest.json

  python scripts/prepare_windowed_parquet.py \\
    --input dataset/processed/AcTBeCalf/test.parquet \\
    --output dataset/processed/AcTBeCalf/windowed_test.parquet \\
    --feature-manifest-in dataset/processed/AcTBeCalf/tsfel_feature_manifest.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tsfel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Residual scipy moment warnings if other stats hit near-constant windows.
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation due to catastrophic cancellation",
    category=RuntimeWarning,
)


LABEL_MAP = {
    "Standing": ["standing"],
    "Lying": ["lying", "lying-down"],
    "Drinking": ["drinking", "drinking_milk", "drinking_electrolytes", "drinking|water"],
    "Eating": ["eating", "eating_concentrates", "eating_bedding", "eating_forage"],
    "Walking": ["walking", "backward"],
    "Run": ["running"],
    "Grooming": ["grooming", "grooming_lying", "grooming|None"],
    "Social Interaction": [
        "social",
        "social_sniff",
        "social_sniff_lying",
        "social_groom",
        "social_groom_lying",
        "social_nudge",
        "social_nudge_lying",
    ],
    "Play": ["play", "play_object", "headbutt", "jump", "mount"],
    "Rising": ["rising"],
    "Rumination": ["rumination", "rumination_lying"],
    "Defecation": ["defecation"],
    "Urination": ["urination"],
    "Oral manipulation of pen": ["oral_manipulation_of_pen"],
    "Sniff": ["sniff", "sniff_walking", "sniff_lying"],
    "Abnormal": [
        "abnormal",
        "cross-suckle_udder",
        "cross-suckle_other",
        "tongue_rolling",
        "tongue_rolling_lying",
    ],
    "SRS": ["SRS", "scratch", "rub", "stretch"],
    "Cough": ["cough"],
    "Fall": ["fall"],
    "Vocalization": ["vocalization", "vocalisation"],
}

RAW_TO_CANONICAL = {
    raw_label.lower().strip(): target_label
    for target_label, raw_labels in LABEL_MAP.items()
    for raw_label in raw_labels
}


def map_behavior(label: str) -> str:
    normalized = str(label).lower().strip()
    return RAW_TO_CANONICAL.get(normalized, "Other")


def tsfel_feature_config() -> dict:
    """TSFEL domain config without skew/kurtosis (statistical and spectral)."""
    cfg = tsfel.get_features_by_domain()
    for domain, name in (
        ("statistical", "Skewness"),
        ("statistical", "Kurtosis"),
        ("spectral", "Spectral skewness"),
        ("spectral", "Spectral kurtosis"),
    ):
        feats = cfg.get(domain)
        if feats is not None and name in feats:
            del feats[name]
    return cfg


def create_windowed_dataframe(
    input_path: Path,
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
) -> pd.DataFrame:
    stride = int(window_size * (1 - overlap))
    cols = [*group_by, time_column, acc_x, acc_y, acc_z, label_column]
    df = pd.read_parquet(input_path, columns=cols)
    df = df.rename(columns={label_column: "_raw_label"})
    df["label"] = df["_raw_label"].map(map_behavior)
    df.drop(columns=["_raw_label"], inplace=True)

    acc_cols = [acc_x, acc_y, acc_z]
    window_list = []
    for key, group in df.groupby(group_by, sort=False):
        if len(group_by) == 1:
            key = (key,)
        raw_data = group[acc_cols].values
        timestamps = group[time_column].values
        labels = group["label"].values
        n_samples = len(raw_data)
        calf_id = key[0]

        if n_samples < window_size:
            continue

        for start_idx in range(0, n_samples - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_labels = labels[start_idx:end_idx]
            counts = Counter(window_labels)
            most_common_label, count = counts.most_common(1)[0]
            if (count / window_size) < purity_threshold:
                continue
            window_signals = raw_data[start_idx:end_idx]
            window_list.append(
                {
                    "dateTime": timestamps[start_idx],
                    "calf_id": calf_id,
                    "acc_x": window_signals[:, 0].tolist(),
                    "acc_y": window_signals[:, 1].tolist(),
                    "acc_z": window_signals[:, 2].tolist(),
                    "label": most_common_label,
                }
            )

    print(f"Janelas geradas: {len(window_list)}")
    return pd.DataFrame(window_list)


def discover_top_features(
    df_windows: pd.DataFrame,
    *,
    top_n: int,
    sample_size: int,
    fs: int,
    random_state: int = 42,
) -> list[str]:
    print("Descoberta de features TSFEL (amostra + RF)...")
    try:
        sample_df, _ = train_test_split(
            df_windows,
            train_size=min(sample_size, len(df_windows)),
            stratify=df_windows["label"],
            random_state=random_state,
        )
    except ValueError:
        sample_df = df_windows.sample(n=min(len(df_windows), sample_size), random_state=random_state)

    print(f"  Amostra: {len(sample_df)} janelas")
    tsfel_input = [
        pd.DataFrame({"accX": row["acc_x"], "accY": row["acc_y"], "accZ": row["acc_z"]})
        for _, row in sample_df.iterrows()
    ]
    cfg = tsfel_feature_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = tsfel.time_series_features_extractor(cfg, tsfel_input, fs=fs, verbose=0)
    X.fillna(0, inplace=True)

    sel_var = VarianceThreshold(threshold=0.02)
    try:
        sel_var.fit(X)
        X = X.loc[:, sel_var.get_support()]
    except Exception:
        pass

    corr_features = tsfel.correlated_features(X, threshold=0.98)
    X.drop(corr_features, axis=1, inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(sample_df["label"])
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
    rf.fit(X, y)
    indices = np.argsort(rf.feature_importances_)[::-1][:top_n]
    top_features = X.columns[indices].tolist()
    print(f"  Top-{len(top_features)} features; ex.: {top_features[:3]}")
    return top_features


def extract_tsfel_batched(
    df_main: pd.DataFrame,
    top_feature_names: list[str],
    *,
    fs: int,
    batch_size: int,
) -> pd.DataFrame:
    tsfel_input_full = [
        pd.DataFrame({"accX": r[0], "accY": r[1], "accZ": r[2]})
        for r in zip(df_main["acc_x"], df_main["acc_y"], df_main["acc_z"])
    ]
    cfg = tsfel_feature_config()
    all_parts = []
    n = len(tsfel_input_full)
    total_batches = (n // batch_size) + (1 if n % batch_size else 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, n, batch_size):
            batch = tsfel_input_full[i : i + batch_size]
            print(f"  TSFEL batch {i // batch_size + 1}/{max(1, total_batches)}...")
            X_batch = tsfel.time_series_features_extractor(cfg, batch, fs=fs, verbose=0)
            X_batch.fillna(0, inplace=True)
            for col in top_feature_names:
                if col not in X_batch.columns:
                    X_batch[col] = 0.0
            all_parts.append(X_batch[top_feature_names])
    return pd.concat(all_parts, axis=0).reset_index(drop=True)


def build_manifest(
    top_feature_names: list[str],
    *,
    window_size: int,
    overlap: float,
    purity_threshold: float,
    fs: int,
    top_n: int,
    sample_size: int,
    group_by: list[str],
    label_column: str,
    acc_x: str,
    acc_y: str,
    acc_z: str,
    time_column: str,
) -> dict:
    return {
        "schema_version": 1,
        "top_feature_names": top_feature_names,
        "window_size": window_size,
        "overlap": overlap,
        "purity_threshold": purity_threshold,
        "fs": fs,
        "top_n": top_n,
        "sample_size": sample_size,
        "group_by": group_by,
        "label_column": label_column,
        "acc_columns": {"x": acc_x, "y": acc_y, "z": acc_z},
        "time_column": time_column,
    }


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "top_feature_names" not in data:
        raise ValueError("Manifest sem top_feature_names")
    if data.get("schema_version", 1) != 1:
        warnings.warn(f"schema_version inesperada: {data.get('schema_version')}")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Parquet bruto → janelado + TSFEL")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature-manifest-out", type=Path, default=None)
    p.add_argument("--feature-manifest-in", type=Path, default=None)
    p.add_argument("--window-size", type=int, default=75)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--purity-threshold", type=float, default=0.9)
    p.add_argument("--fs", type=int, default=25)
    p.add_argument("--top-n", type=int, default=75)
    p.add_argument("--sample-size", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--group-by", nargs="+", default=["calfId", "segId"])
    p.add_argument("--time-column", default="dateTime")
    p.add_argument("--label-column", default="behaviour")
    p.add_argument("--acc-x", dest="acc_x", default="accX")
    p.add_argument("--acc-y", dest="acc_y", default="accY")
    p.add_argument("--acc-z", dest="acc_z", default="accZ")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Entrada não encontrada: {args.input}")

    apply_mode = args.feature_manifest_in is not None
    if apply_mode and not args.feature_manifest_in.is_file():
        raise SystemExit(f"Manifest não encontrado: {args.feature_manifest_in}")

    if apply_mode and args.feature_manifest_out is not None:
        print("Aviso: em modo apply, --feature-manifest-out é ignorado.")

    df_main = create_windowed_dataframe(
        args.input,
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
    if df_main.empty:
        raise SystemExit("Nenhuma janela gerada; verifique dados e hiperparâmetros.")

    if apply_mode:
        manifest = load_manifest(args.feature_manifest_in)
        top_names = manifest["top_feature_names"]
    else:
        top_names = discover_top_features(
            df_main,
            top_n=args.top_n,
            sample_size=args.sample_size,
            fs=args.fs,
        )
        if not args.feature_manifest_out:
            print(
                "Aviso: sem --feature-manifest-out não será possível replicar as mesmas "
                "features no conjunto de teste (modo apply)."
            )
        else:
            man = build_manifest(
                top_names,
                window_size=args.window_size,
                overlap=args.overlap,
                purity_threshold=args.purity_threshold,
                fs=args.fs,
                top_n=args.top_n,
                sample_size=args.sample_size,
                group_by=args.group_by,
                label_column=args.label_column,
                acc_x=args.acc_x,
                acc_y=args.acc_y,
                acc_z=args.acc_z,
                time_column=args.time_column,
            )
            args.feature_manifest_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.feature_manifest_out, "w", encoding="utf-8") as f:
                json.dump(man, f, indent=2)
            print(f"Manifest: {args.feature_manifest_out.resolve()}")

    df_feat = extract_tsfel_batched(
        df_main,
        top_names,
        fs=args.fs,
        batch_size=args.batch_size,
    )
    df_main = df_main.reset_index(drop=True)
    df_final = pd.concat([df_main, df_feat], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(args.output, engine="pyarrow", compression="snappy", index=False)
    print(f"Salvo: {args.output.resolve()} shape={df_final.shape}")


if __name__ == "__main__":
    main()
