"""Deterministic baselines. No fitted parameters, no seed."""

from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_forecast(history: pd.Series, origin: pd.Timestamp, horizon: int = 24) -> np.ndarray:
    """Copy the same hours from the previous day."""
    idx = pd.date_range(origin, periods=horizon, freq="h", tz=origin.tz)
    prev = idx - pd.Timedelta(days=1)
    return history.reindex(prev).to_numpy(dtype=float)


def persistence_forecast(history: pd.Series, origin: pd.Timestamp, horizon: int = 1) -> np.ndarray:
    """Repeat the last observed hour. Diagnostic for 1-step only."""
    last = origin - pd.Timedelta(hours=1)
    value = float(history.reindex([last]).iloc[0])
    return np.repeat(value, horizon)
