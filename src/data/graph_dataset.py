"""Synchronized multi-household windows for graph models (one origin = all N nodes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.splits import TimeSplit
from src.data.windowing import drop_crossing_windows


@dataclass
class GraphWindowBatch:
    x: np.ndarray  # [S, L, N, F]
    future: np.ndarray  # [S, H, C]
    y: np.ndarray  # [S, N, H]
    mask: np.ndarray  # [S, N, H]
    household_ids: np.ndarray  # [N]


def build_graph_windows(
    panel: pd.DataFrame,
    household_ids: list[int],
    split: TimeSplit,
    which: str,
    lookback: int,
    horizon: int,
    history_cols: list[str],
    future_cols: list[str],
) -> GraphWindowBatch:
    """Keep an origin only if every household has finite lookback + future covariates."""
    n = len(household_ids)
    per_house = {}
    common_ts = None
    for hid in household_ids:
        g = panel[panel["household_id"] == hid].sort_values("timestamp").reset_index(drop=True)
        ts = pd.DatetimeIndex(g["timestamp"])
        if common_ts is None:
            common_ts = ts
        elif not ts.equals(common_ts):
            raise RuntimeError("Graph windows require a dense shared timeline across households.")
        per_house[hid] = {
            "hist": g[history_cols].to_numpy(dtype=float),
            "fut": g[future_cols].to_numpy(dtype=float) if future_cols else np.zeros((len(g), 0)),
            "y": pd.to_numeric(g["kwh"], errors="coerce").to_numpy(dtype=float),
            "obs": g["observed"].to_numpy(dtype=bool),
        }
    assert common_ts is not None
    origins = drop_crossing_windows(common_ts, lookback, horizon, split, which)
    index_of = {t: i for i, t in enumerate(common_ts)}

    xs, futs, ys, masks = [], [], [], []
    # Future covariates are town-shared; take them from the first house.
    fut_ref = per_house[household_ids[0]]["fut"]
    for t0 in origins:
        i = index_of[pd.Timestamp(t0)]
        sl_x = slice(i - lookback, i)
        sl_y = slice(i, i + horizon)
        node_x = []
        node_y = []
        node_m = []
        ok = True
        for hid in household_ids:
            x = per_house[hid]["hist"][sl_x]
            if not np.isfinite(x).all():
                ok = False
                break
            node_x.append(x)
            node_y.append(per_house[hid]["y"][sl_y])
            node_m.append(per_house[hid]["obs"][sl_y])
        if not ok:
            continue
        f = fut_ref[sl_y]
        if future_cols and not np.isfinite(f).all():
            continue
        xs.append(np.stack(node_x, axis=1))  # [L, N, F]
        futs.append(f)
        ys.append(np.stack(node_y, axis=0))  # [N, H]
        masks.append(np.stack(node_m, axis=0))
    if not xs:
        raise RuntimeError(f"No valid graph {which} windows for N={n}")
    return GraphWindowBatch(
        x=np.stack(xs),
        future=np.stack(futs),
        y=np.stack(ys),
        mask=np.stack(masks),
        household_ids=np.asarray(household_ids, dtype=np.int64),
    )


class GraphForecastDataset(Dataset):
    def __init__(self, batch: GraphWindowBatch):
        self.x = torch.from_numpy(batch.x).float()
        self.future = torch.from_numpy(batch.future).float()
        self.y = torch.from_numpy(np.nan_to_num(batch.y, nan=0.0)).float()
        self.mask = torch.from_numpy(batch.mask.astype(np.float32))
        self.household_ids = batch.household_ids

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.future[idx], self.y[idx], self.mask[idx]
