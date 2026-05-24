"""Tests for late-fusion ensemble utilities."""

import numpy as np


def test_soft_vote_proba_sums_to_one():
    from hybrid_activity_recognition.training.ensemble import soft_vote_proba

    a = np.array([[0.7, 0.3], [0.4, 0.6]])
    b = np.array([[0.5, 0.5], [0.2, 0.8]])
    out = soft_vote_proba(a, b, weight_a=0.6)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)


def test_align_rf_proba():
    from hybrid_activity_recognition.training.ensemble import align_rf_proba_to_class_indices

    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    classes = np.array([1, 3])
    out = align_rf_proba_to_class_indices(proba, classes, num_classes=4)
    assert out.shape == (2, 4)
    assert out[0, 1] == 0.2
    assert out[0, 3] == 0.8


def test_stacking_fit_predict():
    from hybrid_activity_recognition.training.ensemble import (
        combine_and_predict,
        fit_stacking_logistic,
    )

    rng = np.random.default_rng(0)
    n, c = 40, 5
    pa = rng.random((n, c))
    pa /= pa.sum(axis=1, keepdims=True)
    pb = rng.random((n, c))
    pb /= pb.sum(axis=1, keepdims=True)
    y = rng.integers(0, c, size=n)

    meta = fit_stacking_logistic(pa, pb, y)
    proba, pred = combine_and_predict(pa, pb, y, method="stacking", meta=meta)
    assert proba.shape == (n, c)
    assert pred.shape == (n,)
