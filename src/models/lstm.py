"""Weather-aware LSTM for 24-hour household load.

History encoder sees load + calendar + (optional) past weather.
The horizon head also sees future calendar and oracle ERA5 weather.
A household embedding is what later personalization (P1) will update.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_history: int,
        n_future: int,
        horizon: int,
        n_households: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        household_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=n_history,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.household_embedding = nn.Embedding(n_households, household_embedding_dim)
        fut_dim = n_future * horizon
        self.head = nn.Sequential(
            nn.Linear(hidden_size + household_embedding_dim + fut_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor, future: torch.Tensor, hid: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F], future: [B, H, C], hid: [B]
        _, (h_n, _) = self.lstm(x)
        enc = h_n[-1]
        emb = self.household_embedding(hid)
        fut = future.reshape(future.size(0), -1)
        return self.head(torch.cat([enc, emb, fut], dim=-1))

    def encoder_parameters(self):
        return self.lstm.parameters()

    def head_parameters(self):
        return list(self.head.parameters()) + list(self.household_embedding.parameters())
