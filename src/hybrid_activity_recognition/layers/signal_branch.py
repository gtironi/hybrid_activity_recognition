import torch
import torch.nn as nn


class HybridCNNLSTMSignalBranch(nn.Module):
    """
    Ramo sinal do modelo híbrido didático: CNN 1D + BiLSTM.
    Entrada: (B, 3, 75). Saída: último passo temporal (B, 2 * hidden_lstm).
    """

    def __init__(self, hidden_lstm: int = 64, lstm_layers: int = 2, lstm_dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

    def forward(self, x_signal):
        x = self.cnn(x_signal)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]


class RobustCNNLSTMSignalBranch(nn.Module):
    """
    Ramo sinal mais profundo: três blocos Conv1d + BiLSTM (1 camada).
    Agregação via estados finais forward/backward (h_n).
    """

    def __init__(self, hidden_lstm: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_lstm,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x_signal):
        x = self.cnn(x_signal)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return torch.cat((h_n[-2], h_n[-1]), dim=1)
