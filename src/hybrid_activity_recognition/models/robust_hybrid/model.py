import torch.nn as nn

from hybrid_activity_recognition.layers.fusion import ConcatFusion
from hybrid_activity_recognition.layers.heads import MLPClassificationHead
from hybrid_activity_recognition.layers.signal_branch import RobustCNNLSTMSignalBranch
from hybrid_activity_recognition.layers.tsfel_branch import TsfelMLPBranch


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class RobustHybridModel(nn.Module):
    """
    Variante mais profunda com CNN mais funda e LSTM de 1 camada;
    inicialização Kaiming nas lineares/convs.
    """

    def __init__(self, num_classes: int, n_features_tsfel: int, hidden_lstm: int = 128):
        super().__init__()
        self.signal_branch = RobustCNNLSTMSignalBranch(hidden_lstm=hidden_lstm)
        self.tsfel_branch = TsfelMLPBranch(n_features_tsfel, hidden_dim=128, dropout=0.5)
        self.fusion = ConcatFusion()
        fusion_dim = (hidden_lstm * 2) + 128
        self.classifier_head = MLPClassificationHead(
            fusion_dim, hidden_dim=128, num_classes=num_classes, dropout=0.5
        )
        self.apply(_init_weights)

    def forward(self, x_signal, x_features):
        z_sig = self.signal_branch(x_signal)
        z_ts = self.tsfel_branch(x_features)
        z = self.fusion(z_sig, z_ts)
        return self.classifier_head(z)
