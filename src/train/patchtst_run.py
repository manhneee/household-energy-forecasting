"""E05: global PatchTST with selected weather (same window / scaler path as LSTM)."""

from __future__ import annotations

import pandas as pd

from src.models.patchtst import PatchTSTForecaster
from src.train.lstm_run import _insample_kwh, _model_cfg, _scale_panel
from src.data.dataset import ForecastDataset, build_windows
from src.data.feature_sets import resolve_columns
from src.data.splits import TimeSplit
from src.config import load_weather_config
from src.train.trainer import evaluate_kwh, train_model


def run_patchtst(panel: pd.DataFrame, split: TimeSplit, exp: dict, data_cfg: dict, seed: int) -> dict:
    weather_cfg = load_weather_config()
    weather_set = exp.get("weather_set") or "none"
    if weather_set not in weather_cfg["sets"] and weather_set not in {"none", "temperature", "selected"}:
        raise ValueError(weather_set)
    if exp.get("mode", "global") != "global":
        raise NotImplementedError("E05 is global-only; individual PatchTST is not in the matrix.")

    history_cols, future_cols = resolve_columns(weather_set, panel.columns)
    lookback = int(data_cfg["task"]["lookback_hours"])
    horizon = int(data_cfg["task"]["horizon_hours"])
    cfg = _model_cfg(exp)
    households = sorted(int(x) for x in panel["household_id"].unique())

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
    model = PatchTSTForecaster(
        n_history=len(history_cols),
        n_future=len(future_cols),
        horizon=horizon,
        n_households=len(households),
        lookback=lookback,
        patch_len=int(cfg.get("patch_len", 16)),
        stride=int(cfg.get("stride", 8)),
        d_model=int(cfg.get("d_model", 64)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
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
