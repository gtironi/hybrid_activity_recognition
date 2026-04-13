"""End-to-end dimension and gradient checks for HybridModel via the factory."""

import torch
import pytest

B, T, N_FEATS, N_CLASSES = 4, 75, 120, 19


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _check_forward(model, n_feats=N_FEATS):
    x_sig = torch.randn(B, 3, T)
    x_feat = torch.randn(B, n_feats)
    logits = model(x_sig, x_feat)
    assert logits.shape == (B, N_CLASSES)
    return logits


def _check_gradients(model, n_feats=N_FEATS):
    x_sig = torch.randn(B, 3, T)
    x_feat = torch.randn(B, n_feats)
    logits = model(x_sig, x_feat)
    loss = logits.sum()
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0
    for p in trainable:
        assert p.grad is not None, f"No gradient for param with shape {p.shape}"


@pytest.mark.parametrize("encoder_name", ["cnn_lstm", "robust"])
@pytest.mark.parametrize("input_mode", ["deep_only", "hybrid"])
def test_factory_forward(encoder_name, input_mode):
    from hybrid_activity_recognition.models.modular import build_hybrid_model

    model = build_hybrid_model(
        encoder_name=encoder_name,
        input_mode=input_mode,
        num_classes=N_CLASSES,
        n_tsfel_feats=N_FEATS,
    )
    _check_forward(model)


@pytest.mark.parametrize("encoder_name", ["cnn_lstm", "robust"])
@pytest.mark.parametrize("input_mode", ["deep_only", "hybrid"])
def test_factory_gradients(encoder_name, input_mode):
    from hybrid_activity_recognition.models.modular import build_hybrid_model

    model = build_hybrid_model(
        encoder_name=encoder_name,
        input_mode=input_mode,
        num_classes=N_CLASSES,
        n_tsfel_feats=N_FEATS,
    )
    _check_gradients(model)


@pytest.mark.skipif(not _has_transformers(), reason="transformers not installed")
@pytest.mark.parametrize("input_mode", ["deep_only", "hybrid"])
def test_patchtst_factory(input_mode):
    from hybrid_activity_recognition.models.modular import build_hybrid_model

    model = build_hybrid_model(
        encoder_name="patchtst",
        input_mode=input_mode,
        num_classes=N_CLASSES,
        n_tsfel_feats=N_FEATS,
        context_length=T,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )
    _check_forward(model)


def test_legacy_hybrid_cnn_lstm():
    from hybrid_activity_recognition.models.hybrid_cnn_lstm.model import HybridCNNLSTM

    model = HybridCNNLSTM(num_classes=N_CLASSES, n_features_tsfel=N_FEATS)
    _check_forward(model)


def test_legacy_robust_hybrid():
    from hybrid_activity_recognition.models.robust_hybrid.model import RobustHybridModel

    model = RobustHybridModel(num_classes=N_CLASSES, n_features_tsfel=N_FEATS)
    _check_forward(model)
