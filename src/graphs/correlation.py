"""Train-only Pearson kNN graph.

Geographic edges are impossible on REFIT: no per-house coordinates are published.
This statistical graph is the primary construction. Fit it on train hours only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pearson_knn(
    panel: pd.DataFrame,
    household_ids: list[int],
    k: int = 3,
    split_col: str = "split",
    value_col: str = "kwh",
) -> tuple[np.ndarray, pd.DataFrame]:
    train = panel[panel[split_col] == "train"]
    wide = (
        train.pivot(index="timestamp", columns="household_id", values=value_col)
        .reindex(columns=household_ids)
    )
    corr = wide.corr(method="pearson")
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)

    adj = np.zeros((len(household_ids), len(household_ids)), dtype=np.float64)
    for i, hid in enumerate(household_ids):
        scores = corr.loc[hid].to_numpy()
        # k nearest by absolute correlation; skip the node itself (already 0).
        order = np.argsort(-np.abs(scores))
        keep = [j for j in order if j != i][:k]
        for j in keep:
            adj[i, j] = max(scores[j], 0.0)
    # Symmetric-ize: if either side picked the edge, keep the mean weight.
    adj = np.maximum(adj, adj.T)
    adj = _normalize_adj(adj)
    return adj, corr


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalization D^{-1/2} (A + I) D^{-1/2}."""
    a = adj + np.eye(adj.shape[0])
    deg = a.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d = np.diag(deg_inv_sqrt)
    return d @ a @ d
