from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DenseNetEncoder(nn.Module):
    """
    DenseNet121 feature extractor.
    We set weights=None because checkpoint will load trained weights.
    """

    def __init__(self):
        super().__init__()
        m = torchvision.models.densenet121(weights=None)
        self.features = m.features
        self.out_dim = 1024  # DenseNet121 final feature dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)  # (B, 1024)
        return f


class DenseNetBiLSTM(nn.Module):
    """
    Model expects input:
      x: (B, T, 3, H, W)
    Outputs:
      logits: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int,
        lstm_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()
        self.encoder = DenseNetEncoder()

        self.lstm = nn.LSTM(
            input_size=self.encoder.out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if lstm_layers == 1 else dropout,
        )

        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)             # (B*T, 3, H, W)
        feats = self.encoder(x).view(B, T, -1) # (B, T, 1024)

        out, _ = self.lstm(feats)              # (B, T, hidden*dir)
        last = out[:, -1, :]                   # (B, hidden*dir)
        last = self.dropout(last)
        logits = self.fc(last)                 # (B, num_classes)
        return logits
