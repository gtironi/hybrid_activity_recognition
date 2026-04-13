"""Modular hybrid architecture — factory and public API.

Usage::

    from hybrid_activity_recognition.models.modular import build_hybrid_model

    model = build_hybrid_model(
        encoder_name="robust",
        input_mode="hybrid",
        num_classes=19,
        n_tsfel_feats=120,
    )
"""

from __future__ import annotations

from hybrid_activity_recognition.models.modular.encoders import (
    CNNLSTMEncoder,
    RobustCNNLSTMEncoder,
)
from hybrid_activity_recognition.models.modular.fusion import ConcatFusion
from hybrid_activity_recognition.models.modular.heads import LinearHead, MLPHead
from hybrid_activity_recognition.models.modular.model import HybridModel
from hybrid_activity_recognition.models.modular.tsfel_branches import MLPTsfelBranch

# Legacy name → (encoder_name, input_mode)
_LEGACY_MAP: dict[str, tuple[str, str]] = {
    "hybrid_cnn_lstm": ("cnn_lstm", "hybrid"),
    "robust_hybrid": ("robust", "hybrid"),
}

_ENCODER_REGISTRY: dict[str, type] = {
    "cnn_lstm": CNNLSTMEncoder,
    "robust": RobustCNNLSTMEncoder,
}


def build_hybrid_model(
    encoder_name: str,
    input_mode: str = "hybrid",
    num_classes: int = 19,
    n_tsfel_feats: int = 120,
    head_hidden_dim: int = 256,
    head_dropout: float = 0.4,
    tsfel_hidden_dim: int | None = None,
    tsfel_dropout: float = 0.3,
    **encoder_kwargs,
) -> HybridModel:
    """Build a HybridModel from component names.

    Parameters
    ----------
    encoder_name : str
        ``"cnn_lstm"`` | ``"robust"`` | ``"patchtst"`` (or legacy names).
    input_mode : str
        ``"deep_only"`` or ``"hybrid"``.
    num_classes : int
        Number of output classes.
    n_tsfel_feats : int
        Number of pre-computed TSFEL features (ignored in deep_only mode).
    head_hidden_dim : int
        Hidden dimension of the MLPHead.
    head_dropout : float
        Dropout rate in the MLPHead.
    tsfel_hidden_dim : int | None
        Hidden dimension of the MLPTsfelBranch.  Defaults to ``encoder.output_dim``.
    tsfel_dropout : float
        Dropout rate in the MLPTsfelBranch.
    **encoder_kwargs
        Extra keyword arguments forwarded to the encoder constructor.
    """
    # Resolve legacy names
    if encoder_name in _LEGACY_MAP:
        encoder_name, input_mode = _LEGACY_MAP[encoder_name]

    # Build encoder
    if encoder_name == "patchtst":
        from hybrid_activity_recognition.models.modular.encoders import PatchTSTEncoder

        encoder = PatchTSTEncoder(**encoder_kwargs)
    elif encoder_name in _ENCODER_REGISTRY:
        encoder = _ENCODER_REGISTRY[encoder_name](**encoder_kwargs)
    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name!r}. "
            f"Available: {sorted(list(_ENCODER_REGISTRY) + ['patchtst'])}"
        )

    # Build optional TSFEL branch + fusion
    tsfel_branch = None
    fusion = None
    if input_mode == "hybrid":
        tsfel_hidden = tsfel_hidden_dim if tsfel_hidden_dim is not None else encoder.output_dim
        tsfel_branch = MLPTsfelBranch(n_tsfel_feats, tsfel_hidden, dropout=tsfel_dropout)
        fusion = ConcatFusion(encoder.output_dim, tsfel_branch.output_dim)
        head_in_dim = fusion.output_dim
    elif input_mode == "deep_only":
        head_in_dim = encoder.output_dim
    else:
        raise ValueError(f"Unknown input_mode: {input_mode!r}. Use 'deep_only' or 'hybrid'.")

    head = MLPHead(head_in_dim, head_hidden_dim, num_classes, dropout=head_dropout)

    return HybridModel(encoder, tsfel_branch, fusion, head, input_mode)
