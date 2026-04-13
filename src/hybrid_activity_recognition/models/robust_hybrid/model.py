# TO-DO: remove, it is a legacy file
# Use models.modular.build_hybrid_model(encoder_name="robust", input_mode="hybrid") instead.

from __future__ import annotations

from hybrid_activity_recognition.models.modular import build_hybrid_model


class RobustHybridModel:
    """Legacy wrapper — delegates to the modular architecture."""

    def __new__(cls, num_classes: int, n_features_tsfel: int, hidden_lstm: int = 128):
        return build_hybrid_model(
            encoder_name="robust",
            input_mode="hybrid",
            num_classes=num_classes,
            n_tsfel_feats=n_features_tsfel,
            head_hidden_dim=128,
            head_dropout=0.5,
            tsfel_hidden_dim=128,
            tsfel_dropout=0.5,
            hidden_lstm=hidden_lstm,
        )
