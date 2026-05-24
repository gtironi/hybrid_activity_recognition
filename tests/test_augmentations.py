"""Tests for online signal augmentation."""

import torch


def test_signal_augmentation_changes_signal():
    from hybrid_activity_recognition.data.augmentations import SignalAugmentation

    aug = SignalAugmentation(jitter_std=0.03, scale_std=0.1)
    x = torch.zeros(3, 75)
    y = aug(x.clone())
    assert y.shape == (3, 75)
    assert not torch.allclose(x, y)


def test_calf_dataset_augment_train_only():
    import numpy as np
    from hybrid_activity_recognition.data.dataloader import CalfHybridDataset

    n = 8
    sig = np.random.randn(n, 3, 75).astype(np.float32)
    feat = np.random.randn(n, 10).astype(np.float32)
    labels = np.arange(n) % 3

    subj = np.arange(n, dtype=np.int64) % 2
    ds_aug = CalfHybridDataset(sig, feat, labels, subject_indices=subj, augment=True)
    ds_plain = CalfHybridDataset(sig, feat, labels, augment=False)

    s_aug, _, _, _ = ds_aug[0]
    s_plain, _, _ = ds_plain[0]
    assert s_aug.shape == (3, 75)
    assert not torch.allclose(s_aug, s_plain)
