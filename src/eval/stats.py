"""Wilcoxon across households; Diebold–Mariano only for the final pair."""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_household_mae(mae_a: np.ndarray, mae_b: np.ndarray) -> dict:
    a = np.asarray(mae_a, dtype=float)
    b = np.asarray(mae_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("MAE vectors must align by household")
    diff = a - b
    # N≈17 is low power. Always report the median difference, not only p.
    try:
        result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        pvalue = float(result.pvalue)
        statistic = float(result.statistic)
    except ValueError:
        pvalue = float("nan")
        statistic = float("nan")
    return {
        "n": int(a.size),
        "median_diff_mae": float(np.median(diff)),
        "mean_diff_mae": float(np.mean(diff)),
        "wins_a": int(np.sum(a < b)),
        "wins_b": int(np.sum(b < a)),
        "ties": int(np.sum(a == b)),
        "statistic": statistic,
        "pvalue": pvalue,
    }
