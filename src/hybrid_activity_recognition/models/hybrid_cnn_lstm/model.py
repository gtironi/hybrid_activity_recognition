import torch.nn as nn

from hybrid_activity_recognition.layers.fusion import ConcatFusion
from hybrid_activity_recognition.layers.heads import MLPClassificationHead
from hybrid_activity_recognition.layers.signal_branch import HybridCNNLSTMSignalBranch
from hybrid_activity_recognition.layers.tsfel_branch import TsfelMLPBranch


class HybridCNNLSTM(nn.Module):
    """
    Híbrido didático: ramo sinal (CNN + BiLSTM) + ramo TSFEL (MLP) + fusão + cabeça.
    forward(x_signal, x_features) -> logits (B, num_classes)
    """

    def __init__(self, num_classes: int, n_features_tsfel: int, hidden_lstm: int = 64):
        super().__init__()
        self.signal_branch = HybridCNNLSTMSignalBranch(hidden_lstm=hidden_lstm)
        self.tsfel_branch = TsfelMLPBranch(n_features_tsfel, hidden_dim=64, dropout=0.4)
        self.fusion = ConcatFusion()
        fusion_dim = (hidden_lstm * 2) + 64
        self.classifier_head = MLPClassificationHead(
            fusion_dim, hidden_dim=128, num_classes=num_classes, dropout=0.4
        )

    def forward(self, x_signal, x_features):
        z_sig = self.signal_branch(x_signal)
        z_ts = self.tsfel_branch(x_features)
        z = self.fusion(z_sig, z_ts)
        return self.classifier_head(z)
