"""Dataset for PatchTST self-supervised pretraining (MAE).

Reads a windowed parquet (generated with ``--no-label``) and returns only the
signal tensor ``(C, T)`` — no TSFEL features, no labels.  Z-score normalization
is computed from the training data statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from hybrid_activity_recognition.data.dataloader import _stack_signals


class PretrainWindowDataset(Dataset):
    """Signal-only dataset for self-supervised pretraining.

    Parameters
    ----------
    parquet_path : str | Path
        Path to a windowed parquet that contains ``acc_x, acc_y, acc_z`` columns
        (each cell is a list/array of floats).  ``label`` column is ignored if
        present.
    signal_mean, signal_std : np.ndarray | None
        Per-channel mean/std of shape ``(1, C, 1)`` for z-score normalization.
        If *None*, computed from this parquet (use this for the training set and
        pass the resulting stats to the validation/test sets).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        signal_mean: np.ndarray | None = None,
        signal_std: np.ndarray | None = None,
    ):
        df = pd.read_parquet(parquet_path)
        signals = _stack_signals(df)  # (N, C, T)

        if signal_mean is None or signal_std is None:
            signal_mean = np.mean(signals, axis=(0, 2), keepdims=True)
            signal_std = np.std(signals, axis=(0, 2), keepdims=True)

        self.signal_mean = signal_mean
        self.signal_std = signal_std

        signals = (signals - signal_mean) / (signal_std + 1e-6)
        self.signals = torch.as_tensor(signals, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.signals[idx]


def prepare_pretrain_dataloader(
    parquet_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
    signal_mean: np.ndarray | None = None,
    signal_std: np.ndarray | None = None,
) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """Build a DataLoader for pretraining and return normalization stats."""
    ds = PretrainWindowDataset(parquet_path, signal_mean, signal_std)
    pin_memory = torch.cuda.is_available()
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    return dl, ds.signal_mean, ds.signal_std
