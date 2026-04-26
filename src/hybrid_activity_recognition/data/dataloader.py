from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

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
    val_fraction: float = 0.05,
    parquet_val_path: str | None = None,
    drop_rare_classes_min_count: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, int, int, LabelEncoder]:
    """
    Treino e teste em Parquets distintos (ex.: por sujeito). Ajusta média/desvio do sinal e
    StandardScaler das features TSFEL apenas nas janelas de treino (após split val).
    LabelEncoder fit só no treino; linhas de teste com label desconhecido são removidas (aviso).
    """
    df_train = pd.read_parquet(parquet_train_path)
    if drop_rare_classes_min_count > 0:
        counts = df_train["label"].value_counts()
        rare = counts[counts < drop_rare_classes_min_count].index
        if len(rare):
            df_train = df_train[~df_train["label"].isin(rare)].reset_index(drop=True)

    df_test = pd.read_parquet(parquet_test_path)
    feat_cols = _feature_columns(df_train)
    df_train = _align_tsfel_columns(df_train, feat_cols)
    df_test = _align_tsfel_columns(df_test, feat_cols)

    if parquet_val_path:
        df_val = pd.read_parquet(parquet_val_path)
        df_val = _align_tsfel_columns(df_val, feat_cols)
        df_tr = df_train.reset_index(drop=True)
    else:
        if val_fraction <= 0 or val_fraction >= 1:
            raise ValueError("val_fraction deve estar em ]0, 1[ quando não há parquet_val.")
        indices = np.arange(len(df_train))
        labels_for_split = df_train["label"].values
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            stratify=labels_for_split,
            random_state=random_state,
        )
        df_tr = df_train.iloc[train_idx].reset_index(drop=True)
        df_val = df_train.iloc[val_idx].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(df_tr["label"])

    known = set(le.classes_)
    n_test_before = len(df_test)
    df_test = df_test[df_test["label"].isin(known)].reset_index(drop=True)
    dropped = n_test_before - len(df_test)
    if dropped:
        print(
            f"[dataloader] Teste: removidas {dropped} janelas com label fora do treino "
            f"(classes do encoder: {len(le.classes_)})."
        )

    n_val_before = len(df_val)
    df_val = df_val[df_val["label"].isin(known)].reset_index(drop=True)
    if n_val_before - len(df_val):
        print(
            f"[dataloader] Validação: removidas {n_val_before - len(df_val)} janelas com label desconhecido."
        )

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


def prepare_supervised_dataloaders(
    parquet_path: str,
    batch_size: int = 64,
    num_workers: int = 2,
    random_state: int = 42,
    test_size_first_split: float = 0.2,
    drop_rare_classes_min_count: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, int, int, LabelEncoder]:
    """
    Split 80/10/10 estratificado; normalização de sinal (train stats) e StandardScaler nas features TSFEL.
    Retorna também num_classes e n_feats para construir o modelo.
    """
    df = pd.read_parquet(parquet_path)
    if drop_rare_classes_min_count > 0:
        counts = df["label"].value_counts()
        rare = counts[counts < drop_rare_classes_min_count].index
        if len(rare):
            df = df[~df["label"].isin(rare)].reset_index(drop=True)

    feat_cols = _feature_columns(df)
    signals = _stack_signals(df)
    features = df[feat_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0)

    le = LabelEncoder()
    labels = le.fit_transform(df["label"])
    class_names = le.classes_
    num_classes = len(class_names)
    n_feats = len(feat_cols)

    indices = np.arange(len(df))
    train_idx, temp_idx, _, y_temp = train_test_split(
        indices,
        labels,
        test_size=test_size_first_split,
        stratify=labels,
        random_state=random_state,
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )

    train_signals = signals[train_idx]
    mean_signal = np.mean(train_signals, axis=(0, 2), keepdims=True)
    std_signal = np.std(train_signals, axis=(0, 2), keepdims=True)
    signals_norm = (signals - mean_signal) / (std_signal + 1e-6)

    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    features_norm = scaler.transform(features)

    pin_memory = torch.cuda.is_available()
    train_dl = DataLoader(
        CalfHybridDataset(signals_norm[train_idx], features_norm[train_idx], labels[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        CalfHybridDataset(signals_norm[val_idx], features_norm[val_idx], labels[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dl = DataLoader(
        CalfHybridDataset(signals_norm[test_idx], features_norm[test_idx], labels[test_idx]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dl, val_dl, test_dl, class_names, num_classes, n_feats, le
