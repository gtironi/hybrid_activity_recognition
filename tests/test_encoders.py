"""Dimension sanity checks for all SignalEncoder implementations."""

import torch
import pytest

B, T = 4, 75


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def test_cnn_lstm_encoder_shape():
    from hybrid_activity_recognition.models.encoders import CNNLSTMEncoder

    enc = CNNLSTMEncoder(in_channels=3, hidden_lstm=64)
    x = torch.randn(B, 3, T)
    out = enc(x)
    assert out.shape == (B, enc.output_dim)
    assert enc.output_dim == 128


def test_cnn_lstm_encoder_custom_hidden():
    from hybrid_activity_recognition.models.encoders import CNNLSTMEncoder

    enc = CNNLSTMEncoder(in_channels=3, hidden_lstm=32)
    x = torch.randn(B, 3, T)
    out = enc(x)
    assert out.shape == (B, 64)
    assert enc.output_dim == 64


def test_robust_encoder_shape():
    from hybrid_activity_recognition.models.encoders import RobustCNNLSTMEncoder

    enc = RobustCNNLSTMEncoder(in_channels=3, hidden_lstm=128)
    x = torch.randn(B, 3, T)
    out = enc(x)
    assert out.shape == (B, enc.output_dim)
    assert enc.output_dim == 256


def test_robust_encoder_custom_hidden():
    from hybrid_activity_recognition.models.encoders import RobustCNNLSTMEncoder

    enc = RobustCNNLSTMEncoder(in_channels=3, hidden_lstm=64)
    x = torch.randn(B, 3, T)
    out = enc(x)
    assert out.shape == (B, 128)
    assert enc.output_dim == 128


@pytest.mark.skipif(not _has_transformers(), reason="transformers not installed")
def test_patchtst_encoder_shape():
    from hybrid_activity_recognition.models.encoders import PatchTSTEncoder

    enc = PatchTSTEncoder(context_length=T, d_model=64, num_heads=4, num_layers=2)
    x = torch.randn(B, 3, T)
    out = enc(x)
    assert out.shape == (B, enc.output_dim)
    assert enc.output_dim == 64


def test_null_signal_encoder_shape():
    from hybrid_activity_recognition.models.encoders import NullSignalEncoder

    enc = NullSignalEncoder(output_dim=1)
    x = torch.randn(B, 3, T)
    out = enc(x)
    assert out.shape == (B, 1)
    assert enc.output_dim == 1
