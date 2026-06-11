"""Modular hybrid architecture — factory and public API.

Usage::

    from hybrid_activity_recognition.models import build_hybrid_model

    model = build_hybrid_model(
        encoder_name="cnn_lstm",
        input_mode="hybrid",
        num_classes=19,
        n_tsfel_feats=120,
    )
"""

from __future__ import annotations

from hybrid_activity_recognition.models.encoders import CNNLSTMEncoder
from hybrid_activity_recognition.models.fusion import ConcatFusion
from hybrid_activity_recognition.models.heads import MLPHead
from hybrid_activity_recognition.models.model import HybridModel
from hybrid_activity_recognition.models.tsfel_branches import MLPTsfelBranch

_ENCODER_REGISTRY: dict[str, type] = {
    "cnn_lstm": CNNLSTMEncoder,
}

_TSFEL_BRANCH_REGISTRY: dict[str, type] = {
    "mlp": MLPTsfelBranch,
}


def build_hybrid_model(
    encoder_name: str,
    input_mode: str = "hybrid",
    num_classes: int = 19,
    n_tsfel_feats: int = 120,
    in_channels: int = 3,
    head_name: str = "mlp",
    head_hidden_dim: int = 256,
    head_dropout: float = 0.4,
    tsfel_branch_name: str = "mlp",
    tsfel_hidden_dim: int | None = None,
    tsfel_dropout: float = 0.3,
    **encoder_kwargs,
) -> HybridModel:
    """Build a HybridModel from component names.

    Parameters
    ----------
    encoder_name : str
        ``"cnn_lstm"`` | ``"patchtst"``.
    input_mode : str
        ``"deep_only"`` | ``"hybrid"``.
    num_classes : int
        Number of output classes.
    n_tsfel_feats : int
        Number of pre-computed TSFEL features (ignored in deep_only mode).
    head_hidden_dim : int
        Hidden dimension of the MLPHead.
    head_dropout : float
        Dropout rate in the MLPHead.
    tsfel_branch_name : str
        ``"mlp"`` (identity pass-through). Default ``"mlp"``.
    tsfel_hidden_dim : int | None
        Hidden dimension of the TSFEL branch.  Defaults to ``encoder.output_dim``.
    tsfel_dropout : float
        Dropout rate in the TSFEL branch.
    **encoder_kwargs
        Extra keyword arguments forwarded to the encoder constructor.
    """
    # Build encoder
    if encoder_name == "patchtst":
        from hybrid_activity_recognition.models.encoders import PatchTSTEncoder

        encoder_kwargs.setdefault("in_channels", in_channels)
        encoder = PatchTSTEncoder(**encoder_kwargs)
    elif encoder_name in _ENCODER_REGISTRY:
        encoder_kwargs.setdefault("in_channels", in_channels)
        encoder = _ENCODER_REGISTRY[encoder_name](**encoder_kwargs)
    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name!r}. "
            f"Available: {sorted(list(_ENCODER_REGISTRY) + ['patchtst'])}"
        )

    if tsfel_branch_name not in _TSFEL_BRANCH_REGISTRY:
        raise ValueError(
            f"Unknown tsfel_branch_name: {tsfel_branch_name!r}. "
            f"Available: {sorted(_TSFEL_BRANCH_REGISTRY)}"
        )
    TsfelBranchCls = _TSFEL_BRANCH_REGISTRY[tsfel_branch_name]

    # Build optional TSFEL branch + fusion
    tsfel_branch = None
    fusion = None
    if input_mode == "hybrid":
        tsfel_hidden = tsfel_hidden_dim if tsfel_hidden_dim is not None else encoder.output_dim
        tsfel_branch = TsfelBranchCls(n_tsfel_feats, tsfel_hidden, dropout=tsfel_dropout)
        fusion = ConcatFusion(encoder.output_dim, tsfel_branch.output_dim)
        head_in_dim = fusion.output_dim
    elif input_mode == "deep_only":
        head_in_dim = encoder.output_dim
    else:
        raise ValueError(
            f"Unknown input_mode: {input_mode!r}. Use 'deep_only' or 'hybrid'."
        )

    if head_name == "mlp":
        head = MLPHead(head_in_dim, head_hidden_dim, num_classes, dropout=head_dropout)
    else:
        raise ValueError("Unknown head_name. Use 'mlp'.")

    return HybridModel(encoder, tsfel_branch, fusion, head, input_mode)
