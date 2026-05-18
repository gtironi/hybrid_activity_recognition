"""Dataloaders for AcTBeCalf windowed data (labeled + unlabeled)."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

STANDARD_COLS = ['dateTime', 'calfId', 'calf_id', 'segId', 'acc_x', 'acc_y', 'acc_z', 'label']


class CalfHybridDataset(Dataset):
    def __init__(self, signals, features, labels):
        self.signals = torch.FloatTensor(signals)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]


class UnlabeledDataset(Dataset):
    def __init__(self, signals, features):
        self.signals = torch.FloatTensor(signals)
        self.features = torch.FloatTensor(features)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx]


def prepare_dataloaders(train_parquet_path, test_parquet_path, batch_size=64, num_workers=0):
    def load_and_clean(path):
        df = pd.read_parquet(path)
        counts = df['label'].value_counts()
        to_remove = counts[counts < 2].index
        if len(to_remove) > 0:
            print(f"   Removendo classes <2 amostras: {list(to_remove)}")
            df = df[~df['label'].isin(to_remove)].reset_index(drop=True)
        return df

    print("1. Carregando datasets...")
    df_train = load_and_clean(train_parquet_path)
    df_test = load_and_clean(test_parquet_path)
    print(f"   Train: {len(df_train)} | Test: {len(df_test)}")

    feature_cols = [c for c in df_train.columns if c not in STANDARD_COLS]
    print(f"   Features TSFEL: {len(feature_cols)}")

    def extract(df):
        signals = np.stack([
            np.stack(df['acc_x'].values),
            np.stack(df['acc_y'].values),
            np.stack(df['acc_z'].values),
        ], axis=1).astype(np.float32)
        features = np.nan_to_num(df[feature_cols].values.astype(np.float32))
        return signals, features

    print("2. Extraindo sinais e features...")
    signals_tr, features_tr = extract(df_train)
    signals_te, features_te = extract(df_test)

    print("3. Codificando labels...")
    le = LabelEncoder()
    le.fit(pd.concat([df_train['label'], df_test['label']]))
    labels_tr = le.transform(df_train['label'])
    labels_te = le.transform(df_test['label'])
    class_names = le.classes_
    print(f"   Classes ({len(class_names)}): {list(class_names)}")

    train_idx, val_idx, _, _ = train_test_split(
        np.arange(len(df_train)), labels_tr,
        test_size=0.1, stratify=labels_tr, random_state=42,
    )
    print(f"   Split: train={len(train_idx)} val={len(val_idx)} test={len(labels_te)}")

    print("4. Normalizando...")
    mean_sig = np.mean(signals_tr[train_idx], axis=(0, 2), keepdims=True)
    std_sig = np.std(signals_tr[train_idx], axis=(0, 2), keepdims=True)
    signals_tr_norm = (signals_tr - mean_sig) / (std_sig + 1e-6)
    signals_te_norm = (signals_te - mean_sig) / (std_sig + 1e-6)

    scaler = StandardScaler().fit(features_tr[train_idx])
    features_tr_norm = scaler.transform(features_tr)
    features_te_norm = scaler.transform(features_te)

    print("5. Criando dataloaders...")
    train_ds = CalfHybridDataset(signals_tr_norm[train_idx], features_tr_norm[train_idx], labels_tr[train_idx])
    val_ds = CalfHybridDataset(signals_tr_norm[val_idx], features_tr_norm[val_idx], labels_tr[val_idx])
    test_ds = CalfHybridDataset(signals_te_norm, features_te_norm, labels_te)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl, class_names, len(feature_cols)


def prepare_unlabeled_dataloader(unlabeled_parquet_path, batch_size=64):
    print(f"Carregando unlabeled: {unlabeled_parquet_path}")
    df = pd.read_parquet(unlabeled_parquet_path)
    feat_cols = [c for c in df.columns if c not in STANDARD_COLS]
    print(f"   Unlabeled: {len(df)} amostras | features: {len(feat_cols)}")

    signals = np.stack([
        np.stack(df['acc_x'].values),
        np.stack(df['acc_y'].values),
        np.stack(df['acc_z'].values),
    ], axis=1).astype(np.float32)
    features = np.nan_to_num(df[feat_cols].values.astype(np.float32))

    mean_sig = np.mean(signals, axis=(0, 2), keepdims=True)
    std_sig = np.std(signals, axis=(0, 2), keepdims=True)
    signals = (signals - mean_sig) / (std_sig + 1e-6)
    features = StandardScaler().fit_transform(features)

    ds = UnlabeledDataset(signals, features)
    dl = DataLoader(ds, batch_size=batch_size * 7, shuffle=True, num_workers=0, drop_last=True)
    return dl
