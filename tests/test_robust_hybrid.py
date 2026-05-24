"""Hybrid forward pass with gated fusion and regularized Robust encoder."""

import torch


def test_robust_hybrid_gated_forward():
    from hybrid_activity_recognition.models import build_hybrid_model

    b, t, k, n_cls = 4, 75, 50, 11
    model = build_hybrid_model(
        encoder_name="robust",
        input_mode="hybrid",
        num_classes=n_cls,
        n_tsfel_feats=k,
        fusion_name="gated",
        hidden_lstm=64,
        cnn_dropout=0.4,
        lstm_dropout=0.4,
    )
    x_sig = torch.randn(b, 3, t)
    x_feat = torch.randn(b, k)
    logits = model(x_sig, x_feat)
    assert logits.shape == (b, n_cls)
    loss = logits.sum()
    loss.backward()
