"""Evaluate E01 / E02 on a processed panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import TimeSplit
from src.data.windowing import drop_crossing_windows
from src.eval.metrics import mae as mae_metric
from src.eval.metrics import masked_metrics
from src.models.naive import persistence_forecast, seasonal_naive_forecast


def evaluate_naive(panel: pd.DataFrame, split: TimeSplit, exp: dict, data_cfg: dict) -> dict:
    lookback = data_cfg["task"]["lookback_hours"]
    horizon = int(exp.get("horizon_hours") or data_cfg["task"]["horizon_hours"])
    kind = exp.get("model", {}).get("kind") or exp.get("model")
    y_true_all, y_pred_all, mask_all = [], [], []
    insample = []
    per_house: dict[int, float] = {}
    for hid in sorted(panel["household_id"].unique()):
        hid = int(hid)
        g = panel[panel["household_id"] == hid].sort_values("timestamp")
        values = g.set_index("timestamp")["kwh"].astype(float)
        observed = g.set_index("timestamp")["observed"].astype(bool)
        origins = drop_crossing_windows(
            pd.DatetimeIndex(sorted(values.index)), lookback, horizon, split, "test"
        )
        house_t, house_p, house_m = [], [], []
        for t0 in origins:
            if kind == "persistence" or exp.get("id") == "E02":
                pred = persistence_forecast(values, t0, horizon=horizon)
            else:
                pred = seasonal_naive_forecast(values, t0, horizon=horizon)
            truth_idx = pd.date_range(t0, periods=horizon, freq="h", tz=t0.tz)
            yt = values.reindex(truth_idx).to_numpy(dtype=float)
            ym = observed.reindex(truth_idx).fillna(False).to_numpy(dtype=bool)
            y_true_all.append(yt)
            y_pred_all.append(pred)
            mask_all.append(ym)
            house_t.append(yt)
            house_p.append(pred)
            house_m.append(ym)
        if house_t:
            per_house[hid] = mae_metric(
                np.concatenate(house_t),
                np.concatenate(house_p),
                np.concatenate(house_m),
            )
        train_vals = values[values.index <= split.train_end].to_numpy(dtype=float)
        insample.append(train_vals[np.isfinite(train_vals)])

    metrics = masked_metrics(
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        mask=np.concatenate(mask_all),
        y_insample=np.concatenate(insample) if insample else None,
        seasonality=24,
    )
    metrics["per_house_mae"] = per_house
    return metrics
