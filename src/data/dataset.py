"""Window tensors for the forecast task. One origin = first hour of the 24h horizon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.splits import TimeSplit
from src.data.windowing import drop_crossing_windows


@dataclass
class WindowBatch:
    x: np.ndarray
    future: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    household_id: np.ndarray


def build_windows(
    panel: pd.DataFrame,
    household_ids: list[int],
    split: TimeSplit,
    which: str,
    lookback: int,
    horizon: int,
    history_cols: list[str],
    future_cols: list[str],
) -> WindowBatch:
    """Slice windows by integer offset on the dense hourly grid (no leakage across splits)."""
    xs, futs, ys, masks, hids = [], [], [], [], []
    for hid in household_ids:
        g = panel[panel["household_id"] == hid].sort_values("timestamp").reset_index(drop=True)
        ts = pd.DatetimeIndex(g["timestamp"])
        origins = drop_crossing_windows(ts, lookback, horizon, split, which)
        if origins.empty:
            continue
        # Dense hourly grid: origin timestamp -> row index of first forecast hour.
        index_of = {t: i for i, t in enumerate(ts)}
        hist = g[history_cols].to_numpy(dtype=float)
        fut = g[future_cols].to_numpy(dtype=float) if future_cols else np.zeros((len(g), 0))
        y_raw = pd.to_numeric(g["kwh"], errors="coerce").to_numpy(dtype=float)
        observed = g["observed"].to_numpy(dtype=bool)
        for t0 in origins:
            i = index_of[pd.Timestamp(t0)]
            sl_x = slice(i - lookback, i)
            sl_y = slice(i, i + horizon)
            x = hist[sl_x]
            if not np.isfinite(x).all():
                continue
            f = fut[sl_y]
            if future_cols and not np.isfinite(f).all():
                continue
            xs.append(x)
            futs.append(f)
            ys.append(y_raw[sl_y])
            masks.append(observed[sl_y])
            hids.append(hid)
    if not xs:
        raise RuntimeError(f"No valid {which} windows for households {household_ids}")
    return WindowBatch(
        x=np.stack(xs),
        future=np.stack(futs),
        y=np.stack(ys),
        mask=np.stack(masks),
        household_id=np.asarray(hids, dtype=np.int64),
    )


class ForecastDataset(Dataset):
    def __init__(self, batch: WindowBatch, hid_to_idx: dict[int, int]):
        self.x = torch.from_numpy(batch.x).float()
        self.future = torch.from_numpy(batch.future).float()
        self.y = torch.from_numpy(np.nan_to_num(batch.y, nan=0.0)).float()
        self.mask = torch.from_numpy(batch.mask.astype(np.float32))
        self.hid = torch.tensor([hid_to_idx[int(h)] for h in batch.household_id], dtype=torch.long)
        self.household_id = batch.household_id

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.future[idx], self.y[idx], self.mask[idx], self.hid[idx]
