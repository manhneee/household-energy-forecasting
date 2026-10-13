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


def origins_in_target_span(
    origin: pd.DatetimeIndex,
    lookback: int,
    horizon: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    allow_lookback_before: bool = True,
) -> pd.DatetimeIndex:
    """Keep origins whose *horizon* lies in [target_start, target_end].

    Lookback may start before ``target_start`` (allowed for adaptation: past
    relative to the targets). Inputs must still be after the panel start is
    enforced by the caller via finite-history checks.
    """
    origin = pd.DatetimeIndex(origin)
    last_target = origin + pd.Timedelta(hours=horizon - 1)
    first_input = origin - pd.Timedelta(hours=lookback)
    mask = (origin >= target_start) & (last_target <= target_end)
    if not allow_lookback_before:
        mask = mask & (first_input >= target_start)
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
