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


class UnlabeledDataset(Dataset):
    def __init__(self, signals: np.ndarray, features: np.ndarray):
        self.signals = torch.as_tensor(signals, dtype=torch.float32)
        self.features = torch.as_tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _STANDARD_COLS]


def _stack_signals(df: pd.DataFrame) -> np.ndarray:
    x_stack = np.stack(df["acc_x"].values)
    y_stack = np.stack(df["acc_y"].values)
    z_stack = np.stack(df["acc_z"].values)
    return np.stack([x_stack, y_stack, z_stack], axis=1).astype(np.float32)


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


def prepare_unlabeled_dataloader(
    unlabeled_parquet_path: str,
    batch_size: int = 128,
    num_workers: int = 2,
    unlabeled_batch_multiplier: int = 7,
) -> DataLoader:
    """
    Replica o notebook: normaliza sinal com mean/std globais do array não rotulado;
    StandardScaler ajustado em todas as linhas não rotuladas.
    """
    df_u = pd.read_parquet(unlabeled_parquet_path)
    feat_cols = _feature_columns(df_u)
    signals_u = _stack_signals(df_u)
    features_u = df_u[feat_cols].values.astype(np.float32)
    features_u = np.nan_to_num(features_u, nan=0.0)

    mean_sig = np.mean(signals_u, axis=(0, 2), keepdims=True)
    std_sig = np.std(signals_u, axis=(0, 2), keepdims=True)
    signals_norm = (signals_u - mean_sig) / (std_sig + 1e-6)

    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features_u)

    pin_memory = torch.cuda.is_available()
    ds = UnlabeledDataset(signals_norm, features_norm)
    return DataLoader(
        ds,
        batch_size=batch_size * max(1, unlabeled_batch_multiplier),
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
