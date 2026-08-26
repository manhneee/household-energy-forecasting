"""Masked metrics in physical kWh. Never report scaled units. Do not lead with MAPE."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if mask is None:
        keep = np.isfinite(y_true) & np.isfinite(y_pred)
    else:
        keep = np.asarray(mask, dtype=bool).ravel() & np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[keep], y_pred[keep]


def mae(y_true, y_pred, mask=None) -> float:
    yt, yp = _finite_pair(y_true, y_pred, mask)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred, mask=None) -> float:
    yt, yp = _finite_pair(y_true, y_pred, mask)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def smape(y_true, y_pred, mask=None) -> float:
    yt, yp = _finite_pair(y_true, y_pred, mask)
    if yt.size == 0:
        return float("nan")
    denom = np.abs(yt) + np.abs(yp)
    ok = denom > 0
    if not np.any(ok):
        return float("nan")
    return float(100.0 * np.mean(2.0 * np.abs(yt[ok] - yp[ok]) / denom[ok]))


def wape(y_true, y_pred, mask=None) -> float:
    yt, yp = _finite_pair(y_true, y_pred, mask)
    if yt.size == 0 or np.sum(np.abs(yt)) == 0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / np.sum(np.abs(yt)))


def mase(y_true, y_pred, y_insample: np.ndarray, seasonality: int = 24, mask=None) -> float:
    """MASE scaled by the in-sample seasonal naive MAE (Hyndman & Koehler)."""
    yt, yp = _finite_pair(y_true, y_pred, mask)
    ins = np.asarray(y_insample, dtype=float)
    ins = ins[np.isfinite(ins)]
    if yt.size == 0 or ins.size <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(ins[seasonality:] - ins[:-seasonality]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / scale)


def masked_metrics(
    y_true,
    y_pred,
    mask=None,
    y_insample=None,
    seasonality: int = 24,
) -> dict[str, float]:
    out = {
        "mae": mae(y_true, y_pred, mask),
        "rmse": rmse(y_true, y_pred, mask),
        "smape": smape(y_true, y_pred, mask),
        "wape": wape(y_true, y_pred, mask),
    }
    if y_insample is not None:
        out["mase"] = mase(y_true, y_pred, y_insample, seasonality=seasonality, mask=mask)
    return out


def per_household_mae(frame: pd.DataFrame, truth="y_true", pred="y_pred", mask="observed") -> pd.Series:
    rows = {}
    for hid, g in frame.groupby("household_id"):
        m = g[mask].to_numpy() if mask in g.columns else None
        rows[int(hid)] = mae(g[truth], g[pred], m)
    return pd.Series(rows, name="mae")
