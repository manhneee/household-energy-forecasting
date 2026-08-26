"""Chronological 70/15/15 split on the common timeline.

Windows whose horizon crosses a boundary are the caller's problem
(see windowing.drop_crossing_windows). Scalers and graphs fit on train only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp
    start: pd.Timestamp


def chronological_split(
    timestamps: pd.DatetimeIndex,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> TimeSplit:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1")
    ts = pd.DatetimeIndex(sorted(timestamps.unique()))
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    n = len(ts)
    if n < 10:
        raise ValueError(f"Not enough timestamps to split: {n}")
    i_train = int(n * train_frac) - 1
    i_val = int(n * (train_frac + val_frac)) - 1
    i_train = max(0, min(i_train, n - 3))
    i_val = max(i_train + 1, min(i_val, n - 2))
    return TimeSplit(
        start=ts[0],
        train_end=ts[i_train],
        val_end=ts[i_val],
        test_end=ts[-1],
    )


def assign_split(timestamp: pd.Series, split: TimeSplit) -> pd.Series:
    out = pd.Series("test", index=timestamp.index, dtype="object")
    out[timestamp <= split.train_end] = "train"
    out[(timestamp > split.train_end) & (timestamp <= split.val_end)] = "val"
    return out


def adaptation_window(split: TimeSplit, k_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Last K days immediately before the test period. Never overlaps test."""
    if k_days < 1:
        raise ValueError("k_days must be >= 1")
    adapt_end = split.val_end
    adapt_start = adapt_end - pd.Timedelta(days=k_days) + pd.Timedelta(hours=1)
    if adapt_start < split.start:
        adapt_start = split.start
    if adapt_end >= split.val_end + pd.Timedelta(hours=1):
        raise RuntimeError("Adaptation window leaked into test")
    return adapt_start, adapt_end
