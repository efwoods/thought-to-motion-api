import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Model definitions
class EcogMotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# CNN/LSTM hybrid
class EcogToMotionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN component: outputs 256 channels
        self.convolv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),  # Fixed to 256 channels
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        # Bi-LSTM component (2 Layers)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_weight = nn.Linear(2 * 128, 1, bias=False)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(2 * 128, 3),  # Matches hidden_size=128
        )

    def forward(self, x):
        # Input shape: (batch, 20, 64)
        x = x.permute(0, 2, 1)  # Shape: (batch, 64, 20)
        x = self.convolv(x)  # Shape: (batch, 256, 20)
        x = x.permute(0, 2, 1)  # Shape: (batch, 20, 256)

        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out shape: (batch, 20, 128)

        # Compute attention scores
        # Flatten across features: attn_score[i, t] = wT * h_{i, t}
        # Then softmax over t to get α_{i, t}
        attn_scores = self.attn_weight(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        # Weighted sum of LSTM outputs:
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        # Regression to 3D motion
        output = self.fc(attn_applied)
        return output
