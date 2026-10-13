"""Weather-aware GCGRU / T-GCN. Node = household; edges = train-only Pearson kNN."""

from __future__ import annotations

import torch
from torch import nn


class GCGRUCell(nn.Module):
    """T-GCN-style cell: graph-filter the input, then GRU update per node."""

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gcn = nn.Linear(in_features, hidden_size, bias=False)
        self.gates = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.candidate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], h: [B, N, H], adj: [N, N] (already D^{-1/2}(A+I)D^{-1/2})
        gx = torch.matmul(adj, self.gcn(x))
        ru = torch.sigmoid(self.gates(torch.cat([gx, h], dim=-1)))
        r, u = ru.chunk(2, dim=-1)
        c = torch.tanh(self.candidate(torch.cat([gx, r * h], dim=-1)))
        return u * h + (1.0 - u) * c


class GCGRUForecaster(nn.Module):
    """Encode lookback with GCGRU; future calendar/weather enter the MLP head only."""

    def __init__(
        self,
        n_history: int,
        n_future: int,
        horizon: int,
        n_households: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_households = n_households
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        cells = []
        for layer in range(num_layers):
            in_dim = n_history if layer == 0 else hidden_size
            cells.append(GCGRUCell(in_dim, hidden_size))
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)
        fut_dim = n_future * horizon
        self.head = nn.Sequential(
            nn.Linear(hidden_size + fut_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )
        # Filled by set_adjacency before training; buffer so it moves with .to(device).
        self.register_buffer("adj", torch.eye(n_households))

    def set_adjacency(self, adj: torch.Tensor | object) -> None:
        if not torch.is_tensor(adj):
            adj = torch.as_tensor(adj, dtype=torch.float32)
        if adj.shape != (self.n_households, self.n_households):
            raise ValueError(f"adj shape {tuple(adj.shape)} != ({self.n_households}, {self.n_households})")
        self.adj.copy_(adj.to(dtype=self.adj.dtype, device=self.adj.device))

    def forward(self, x: torch.Tensor, future: torch.Tensor, hid: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, N, F], future: [B, H, C] (town weather/calendar, shared), optional hid unused
        del hid
        b, lookback, n, _ = x.shape
        if n != self.n_households:
            raise ValueError(f"got N={n}, expected {self.n_households}")
        states = [
            x.new_zeros(b, n, self.hidden_size) for _ in range(self.num_layers)
        ]
        for t in range(lookback):
            inp = x[:, t]
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(inp, states[layer], self.adj)
                inp = self.dropout(states[layer])
        enc = states[-1]
        fut = future.reshape(b, -1).unsqueeze(1).expand(-1, n, -1)
        return self.head(torch.cat([enc, fut], dim=-1))  # [B, N, horizon]

    def encoder_parameters(self):
        return self.cells.parameters()

    def head_parameters(self):
        return self.head.parameters()


def build_gcgru(**kwargs) -> GCGRUForecaster:
    return GCGRUForecaster(**kwargs)
