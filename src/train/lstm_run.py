"""E03 / E03b / E04 / E04b: individual vs global LSTM, with or without weather."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_weather_config
from src.data.dataset import ForecastDataset, build_windows
from src.data.feature_sets import resolve_columns
from src.data.scalers import ColumnScaler, HouseLoadScaler
from src.data.splits import TimeSplit
from src.eval.metrics import masked_metrics
from src.models.lstm import LSTMForecaster
from src.train.trainer import evaluate_kwh, predict_dataset, train_model


def _scale_panel(panel: pd.DataFrame, history_cols: list[str], future_cols: list[str]) -> tuple[pd.DataFrame, HouseLoadScaler]:
    load_scaler = HouseLoadScaler().fit(panel)
    work = panel.copy()
    work["kwh"] = load_scaler.transform_column(work, "kwh")
    work["kwh_input"] = load_scaler.transform_column(work, "kwh_input")
    weather_cols = [c for c in set(history_cols + future_cols) if c not in {"kwh_input"} and not c.startswith("hour") and not c.startswith("dow") and not c.startswith("month") and not c.startswith("is_")]
    if weather_cols:
        work = ColumnScaler().fit(panel, weather_cols).transform(work, weather_cols)
    return work, load_scaler


def _insample_kwh(panel: pd.DataFrame) -> np.ndarray:
    train = panel[(panel["split"] == "train") & panel["observed"]]
    return pd.to_numeric(train["kwh"], errors="coerce").to_numpy(dtype=float)


def _model_cfg(exp: dict) -> dict:
    return exp["model"] if isinstance(exp.get("model"), dict) else {}


def run_lstm(panel: pd.DataFrame, split: TimeSplit, exp: dict, data_cfg: dict, seed: int) -> dict:
    weather_cfg = load_weather_config()
    weather_set = exp.get("weather_set") or "none"
    if weather_set not in weather_cfg["sets"] and weather_set not in {"none", "temperature", "selected"}:
        raise ValueError(weather_set)
    history_cols, future_cols = resolve_columns(weather_set, panel.columns)
    lookback = data_cfg["task"]["lookback_hours"]
    horizon = data_cfg["task"]["horizon_hours"]
    cfg = _model_cfg(exp)
    mode = exp.get("mode", "global")
    households = sorted(int(x) for x in panel["household_id"].unique())

    if mode == "individual":
        return _run_individual(
            panel, split, households, history_cols, future_cols, lookback, horizon, cfg, seed
        )
    return _run_global(panel, split, households, history_cols, future_cols, lookback, horizon, cfg, seed)


def _run_global(panel, split, households, history_cols, future_cols, lookback, horizon, cfg, seed) -> dict:
    scaled, scaler = _scale_panel(panel, history_cols, future_cols)
    hid_to_idx = {h: i for i, h in enumerate(households)}
    train_ds = ForecastDataset(
        build_windows(scaled, households, split, "train", lookback, horizon, history_cols, future_cols),
        hid_to_idx,
    )
    val_ds = ForecastDataset(
        build_windows(scaled, households, split, "val", lookback, horizon, history_cols, future_cols),
        hid_to_idx,
    )
    test_ds = ForecastDataset(
        build_windows(scaled, households, split, "test", lookback, horizon, history_cols, future_cols),
        hid_to_idx,
    )
    model = LSTMForecaster(
        n_history=len(history_cols),
        n_future=len(future_cols),
        horizon=horizon,
        n_households=len(households),
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        household_embedding_dim=int(cfg.get("household_embedding_dim", 8)),
    )
    insample = _insample_kwh(panel)
    model, train_info = train_model(
        model,
        train_ds,
        val_ds,
        scaler,
        lr=float(cfg.get("lr", 1e-3)),
        batch_size=int(cfg.get("batch_size", 64)),
        max_epochs=int(cfg.get("max_epochs", 40)),
        patience=int(cfg.get("patience", 8)),
        seed=seed,
        y_insample=insample,
    )
    metrics = evaluate_kwh(model, test_ds, scaler, int(cfg.get("batch_size", 64)), y_insample=insample)
    metrics.update(train_info)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    metrics["n_households"] = len(households)
    return metrics


def _run_individual(panel, split, households, history_cols, future_cols, lookback, horizon, cfg, seed) -> dict:
    preds, trues, masks = [], [], []
    train_seconds = 0.0
    n_params = 0
    for hid in households:
        sub = panel[panel["household_id"] == hid].copy()
        scaled, scaler = _scale_panel(sub, history_cols, future_cols)
        hid_to_idx = {hid: 0}
        train_ds = ForecastDataset(
            build_windows(scaled, [hid], split, "train", lookback, horizon, history_cols, future_cols),
            hid_to_idx,
        )
        val_ds = ForecastDataset(
            build_windows(scaled, [hid], split, "val", lookback, horizon, history_cols, future_cols),
            hid_to_idx,
        )
        test_ds = ForecastDataset(
            build_windows(scaled, [hid], split, "test", lookback, horizon, history_cols, future_cols),
            hid_to_idx,
        )
        model = LSTMForecaster(
            n_history=len(history_cols),
            n_future=len(future_cols),
            horizon=horizon,
            n_households=1,
            hidden_size=int(cfg.get("hidden_size", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
            household_embedding_dim=int(cfg.get("household_embedding_dim", 8)),
        )
        insample = _insample_kwh(sub)
        model, info = train_model(
            model,
            train_ds,
            val_ds,
            scaler,
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=int(cfg.get("batch_size", 64)),
            max_epochs=int(cfg.get("max_epochs", 40)),
            patience=int(cfg.get("patience", 8)),
            seed=seed,
            y_insample=insample,
        )
        pred_s, true_s, mask, hids = predict_dataset(model, test_ds, int(cfg.get("batch_size", 64)))
        hid_h = np.repeat(hids, pred_s.shape[1])
        preds.append(scaler.inverse(pred_s.ravel(), hid_h).reshape(pred_s.shape))
        trues.append(scaler.inverse(true_s.ravel(), hid_h).reshape(true_s.shape))
        masks.append(mask)
        train_seconds += info["train_seconds"]
        n_params = sum(p.numel() for p in model.parameters())
    metrics = masked_metrics(
        np.concatenate(trues),
        np.concatenate(preds),
        mask=np.concatenate(masks),
        y_insample=_insample_kwh(panel),
        seasonality=24,
    )
    metrics["train_seconds"] = train_seconds
    metrics["n_params"] = n_params
    metrics["n_households"] = len(households)
    return metrics
