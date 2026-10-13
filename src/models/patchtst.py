"""PatchTST global baseline (Nie et al.): patch lookback, Transformer encoder, weather in head."""

from __future__ import annotations

import torch
from torch import nn


class PatchTSTForecaster(nn.Module):
    """Multivariate time-patch Transformer with future covariates in the MLP head.

    History ``x`` is cut into overlapping patches along time; each patch (all
    channels flattened) is projected to ``d_model``. Future calendar / ERA5
    enter the head only — same contract as ``LSTMForecaster``.
    """

    def __init__(
        self,
        n_history: int,
        n_future: int,
        horizon: int,
        n_households: int,
        lookback: int = 168,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        household_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if lookback < patch_len:
            raise ValueError(f"lookback {lookback} < patch_len {patch_len}")
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len * n_history, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.household_embedding = nn.Embedding(n_households, household_embedding_dim)
        fut_dim = n_future * horizon
        self.head = nn.Sequential(
            nn.Linear(d_model + household_embedding_dim + fut_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor, future: torch.Tensor, hid: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F], future: [B, H, C], hid: [B]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B, n_patches, F, patch_len] -> [B, n_patches, F * patch_len]
        b, n, f, p = patches.shape
        if n != self.n_patches:
            raise ValueError(f"expected {self.n_patches} patches, got {n} (L={x.size(1)})")
        tokens = self.patch_proj(patches.reshape(b, n, f * p)) + self.pos
        enc = self.encoder(tokens).mean(dim=1)
        emb = self.household_embedding(hid)
        fut = future.reshape(future.size(0), -1)
        return self.head(torch.cat([enc, emb, fut], dim=-1))

    def encoder_parameters(self):
        return list(self.patch_proj.parameters()) + list(self.encoder.parameters()) + [self.pos]

    def head_parameters(self):
        return list(self.head.parameters()) + list(self.household_embedding.parameters())


def build_patchtst(**kwargs) -> PatchTSTForecaster:
    return PatchTSTForecaster(**kwargs)
