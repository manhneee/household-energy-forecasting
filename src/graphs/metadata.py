"""Similarity graph from Murray et al. (2017) Table 2.

This is the realistic substitute for a geographic graph on REFIT.
Features are static, so there is no temporal leakage, but the graph is still
restricted to the households that survived the quality protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.data.metadata import numeric_features
from src.graphs.correlation import _normalize_adj


def metadata_similarity(household_ids: list[int], k: int = 3) -> np.ndarray:
    feats = numeric_features(household_ids)
    # Median-fill dwelling_year for House 2 (unpublished age).
    feats = feats.fillna(feats.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(feats.to_numpy(dtype=float))
    sim = cosine_similarity(scaled)
    np.fill_diagonal(sim, 0.0)
    adj = np.zeros_like(sim)
    for i in range(len(household_ids)):
        keep = np.argsort(-sim[i])[:k]
        adj[i, keep] = np.clip(sim[i, keep], 0, None)
    adj = np.maximum(adj, adj.T)
    return _normalize_adj(adj)
