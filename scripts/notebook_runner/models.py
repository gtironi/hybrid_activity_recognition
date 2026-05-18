"""Model architectures: TsfelOnly, RobustHybrid, HybridCNNLSTM."""
import torch
import torch.nn as nn


class HybridCNNLSTM(nn.Module):
    def __init__(self, num_classes, n_features_tsfel, hidden_lstm=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_lstm, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.mlp_tsfel = nn.Sequential(
            nn.Linear(n_features_tsfel, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Linear((hidden_lstm * 2) + 64, 128),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_signal, x_features):
        x_cnn = self.cnn(x_signal).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_cnn)
        x_temporal = lstm_out[:, -1, :]
        x_static = self.mlp_tsfel(x_features)
        return self.classifier(torch.cat((x_temporal, x_static), dim=1))


class RobustHybridModel(nn.Module):
    def __init__(self, num_classes, n_features_tsfel, hidden_lstm=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, hidden_lstm, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.mlp_tsfel = nn.Sequential(
            nn.Linear(n_features_tsfel, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
        )
        fusion_dim = (hidden_lstm * 2) + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_signal, x_features):
        feats = self.cnn(x_signal).permute(0, 2, 1)
        _, (h_n, _) = self.lstm(feats)
        temporal = torch.cat((h_n[-2], h_n[-1]), dim=1)
        static = self.mlp_tsfel(x_features)
        return self.classifier(torch.cat((temporal, static), dim=1))


class TsfelOnlyModel(nn.Module):
    def __init__(self, num_classes, n_features_tsfel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features_tsfel, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(64, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_signal, x_features):
        return self.classifier(self.mlp(x_features))
