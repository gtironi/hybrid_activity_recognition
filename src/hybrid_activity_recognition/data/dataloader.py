from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import genSplit

# Colunas meta comuns entre pipelines (window_creator_* e notebook)
_STANDARD_COLS = frozenset(
    {"dateTime", "calfId", "calf_id", "segId", "acc_x", "acc_y", "acc_z", "label"}
)


class CalfHybridDataset(Dataset):
    def __init__(self, signals: np.ndarray, features: np.ndarray, labels: np.ndarray):
        self.signals = torch.as_tensor(signals, dtype=torch.float32)
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _STANDARD_COLS]


def _subject_behavior_wide(df: pd.DataFrame, subject_col: str, behavior_col: str) -> pd.DataFrame:
    g = df.groupby([subject_col, behavior_col], sort=False).size().rename("n").reset_index()
    wide = g.pivot(index=subject_col, columns=behavior_col, values="n").fillna(0).astype(np.int64)
    return wide.reset_index().rename(columns={subject_col: "subject_id"})


def _split_train_val_by_subject_gen_split(
    df_train: pd.DataFrame,
    *,
    subject_column: str,
    label_column: str,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction deve estar em ]0, 1[ quando não há parquet_val.")

    if subject_column not in df_train.columns:
        raise ValueError(f"Coluna de sujeito ausente: {subject_column!r}")

    wide = _subject_behavior_wide(df_train, subject_column, label_column)
    subjects = sorted(wide["subject_id"].unique().tolist(), key=str)
    n_sub = len(subjects)
    if n_sub < 2:
        raise ValueError("São necessários pelo menos 2 sujeitos no treino para criar validação.")

    val_pct = round(100.0 * val_fraction)
    train_pct = 100 - val_pct
    if train_pct + val_pct != 100 or val_pct <= 0:
        raise ValueError(
            f"val_fraction inválido para split por sujeitos: {val_fraction!r}. "
            "Escolha um valor que produza porcentagens inteiras para treino/validação."
        )

    n_tr, n_val, n_te = genSplit.calc_split_subject_amounts(
        n_sub, {"train": float(train_pct), "validation": float(val_pct), "test": 0.0}
    )
    if n_te != 0 or n_val < 1 or n_tr < 1:
        raise ValueError(f"Contagem inválida de sujeitos: train={n_tr} val={n_val} test={n_te} (n={n_sub}).")

    split_ratio = val_fraction / (1.0 - val_fraction)
    val_tuple = genSplit.find_optimal_calf_combinations_for_split(
        tuple(subjects), n_val, wide, split_ratio, cv=1
    )
    if val_tuple is None:
        raise ValueError(
            "genSplit não encontrou combinação válida para validação; "
            "todas as classes precisam permanecer no treino."
        )

    val_ids = set(val_tuple)
    mask = df_train[subject_column].isin(val_ids)
    df_tr = df_train.loc[~mask].reset_index(drop=True)
    df_val = df_train.loc[mask].reset_index(drop=True)
    return df_tr, df_val


def _stack_signals(df: pd.DataFrame) -> np.ndarray:
    x_stack = np.stack(df["acc_x"].values)
    y_stack = np.stack(df["acc_y"].values)
    z_stack = np.stack(df["acc_z"].values)
    return np.stack([x_stack, y_stack, z_stack], axis=1).astype(np.float32)


def _align_tsfel_columns(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Garante as mesmas colunas TSFEL que no treino (ausentes → 0)."""
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out


def prepare_train_val_test_loaders(
    parquet_train_path: str,
    parquet_test_path: str,
    batch_size: int = 64,
    num_workers: int = 2,
    random_state: int = 42,
    val_fraction: float = 0.2,
    parquet_val_path: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, int, int, LabelEncoder]:
    """
    Treino e teste em Parquets distintos (ex.: por sujeito). Ajusta média/desvio do sinal e
    StandardScaler das features TSFEL apenas nas janelas de treino (após split val).
    LabelEncoder fit só no treino. Labels treino/teste devem estar alinhados em
    scripts/dataset_processing.py (classes raras removidas lá, não aqui); labels desconhecidos
    em teste/val geram erro explícito.
    """
    df_train = pd.read_parquet(parquet_train_path)
    df_train["label"] = df_train["label"].astype(str)

    df_test = pd.read_parquet(parquet_test_path)
    df_test["label"] = df_test["label"].astype(str)
    feat_cols = _feature_columns(df_train)
    df_train = _align_tsfel_columns(df_train, feat_cols)
    df_test = _align_tsfel_columns(df_test, feat_cols)

    if parquet_val_path:
        df_val = pd.read_parquet(parquet_val_path)
        df_val["label"] = df_val["label"].astype(str)
        df_val = _align_tsfel_columns(df_val, feat_cols)
        df_tr = df_train.reset_index(drop=True)
    else:
        subject_col = "calf_id" if "calf_id" in df_train.columns else "calfId"
        df_tr, df_val = _split_train_val_by_subject_gen_split(
            df_train,
            subject_column=subject_col,
            label_column="label",
            val_fraction=val_fraction,
        )

    le = LabelEncoder()
    le.fit(df_tr["label"])
    known = set(le.classes_)

    unk_te = set(df_test["label"].unique()) - known
    if unk_te:
        raise ValueError(
            "Test parquet contains labels not in the training split: "
            f"{sorted(unk_te)!r}. Regenerate parquets with scripts/dataset_processing.py or align labels."
        )
    if df_test.empty:
        raise ValueError("Conjunto de teste está vazio.")

    unk_val = set(df_val["label"].unique()) - known
    if unk_val:
        raise ValueError(
            "Validation set contains labels not in the training split: "
            f"{sorted(unk_val)!r}. Check parquet_val or val_fraction split."
        )
    if df_val.empty:
        raise ValueError("Conjunto de validacao está vazio.")

    signals_tr = _stack_signals(df_tr)
    signals_val = _stack_signals(df_val)
    signals_te = _stack_signals(df_test)
    features_tr = df_tr[feat_cols].values.astype(np.float32)
    features_val = df_val[feat_cols].values.astype(np.float32)
    features_te = df_test[feat_cols].values.astype(np.float32)
    features_tr = np.nan_to_num(features_tr, nan=0.0)
    features_val = np.nan_to_num(features_val, nan=0.0)
    features_te = np.nan_to_num(features_te, nan=0.0)

    y_tr = le.transform(df_tr["label"])
    y_val = le.transform(df_val["label"])
    y_te = le.transform(df_test["label"])

    mean_signal = np.mean(signals_tr, axis=(0, 2), keepdims=True)
    std_signal = np.std(signals_tr, axis=(0, 2), keepdims=True)

    def _norm_sig(s: np.ndarray) -> np.ndarray:
        return (s - mean_signal) / (std_signal + 1e-6)

    signals_tr_n = _norm_sig(signals_tr)
    signals_val_n = _norm_sig(signals_val)
    signals_te_n = _norm_sig(signals_te)

    scaler = StandardScaler()
    scaler.fit(features_tr)
    features_tr_n = scaler.transform(features_tr)
    features_val_n = scaler.transform(features_val)
    features_te_n = scaler.transform(features_te)

    class_names = le.classes_
    num_classes = len(class_names)
    n_feats = len(feat_cols)

    pin_memory = torch.cuda.is_available()
    train_dl = DataLoader(
        CalfHybridDataset(signals_tr_n, features_tr_n, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        CalfHybridDataset(signals_val_n, features_val_n, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dl = DataLoader(
        CalfHybridDataset(signals_te_n, features_te_n, y_te),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dl, val_dl, test_dl, class_names, num_classes, n_feats, le
