"""Sliding windows. A window is kept only if its horizon stays inside one split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import TimeSplit


def drop_crossing_windows(
    origin: pd.DatetimeIndex,
    lookback: int,
    horizon: int,
    split: TimeSplit,
    which: str,
) -> pd.DatetimeIndex:
    """`origin` is the first forecast hour (lookback just ended)."""
    origin = pd.DatetimeIndex(origin)
    last_target = origin + pd.Timedelta(hours=horizon - 1)
    first_input = origin - pd.Timedelta(hours=lookback)
    if which == "train":
        mask = (first_input >= split.start) & (last_target <= split.train_end)
    elif which == "val":
        mask = (origin > split.train_end) & (last_target <= split.val_end)
    elif which == "test":
        mask = (origin > split.val_end) & (last_target <= split.test_end)
    else:
        raise ValueError(which)
    return origin[mask]


def extract_windows(
    series: pd.Series,
    origins: pd.DatetimeIndex,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [n, lookback], y [n, horizon]) in the series' native units."""
    values = series.astype(float)
    x_list = []
    y_list = []
    for t0 in origins:
        x_idx = pd.date_range(t0 - pd.Timedelta(hours=lookback), t0 - pd.Timedelta(hours=1), freq="h", tz=t0.tz)
        y_idx = pd.date_range(t0, t0 + pd.Timedelta(hours=horizon - 1), freq="h", tz=t0.tz)
        x = values.reindex(x_idx).to_numpy()
        y = values.reindex(y_idx).to_numpy()
        x_list.append(x)
        y_list.append(y)
    return np.stack(x_list), np.stack(y_list)
