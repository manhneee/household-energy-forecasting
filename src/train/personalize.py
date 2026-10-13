"""E08 few-shot personalization of a pretrained global LSTM (E04b protocol).

P0 frozen | P1 head+embedding | P2 full fine-tune (lr = 1/10 pretrain).
Adaptation = last K days before test; early stop on last 20% of that window.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_weather_config
from src.data.dataset import ForecastDataset, build_windows, build_windows_in_span
from src.data.feature_sets import resolve_columns
from src.data.splits import TimeSplit, adaptation_window
from src.eval.metrics import masked_metrics, mae as mae_metric
from src.models.lstm import LSTMForecaster
from src.train.lstm_run import _insample_kwh, _model_cfg, _scale_panel
from src.train.seeds import set_seed
from src.train.trainer import evaluate_kwh, make_loader, masked_mse, predict_dataset, train_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _adapt_time_split(adapt_start: pd.Timestamp, adapt_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (adapt_start, early_stop_start, adapt_end). Last 20% is early-stop only."""
    span = adapt_end - adapt_start
    es_start = adapt_end - span * 0.2
    if es_start <= adapt_start:
        es_start = adapt_start + (span * 0.5 if span.total_seconds() > 0 else pd.Timedelta(hours=1))
    return adapt_start, es_start, adapt_end


def _try_span_dataset(
    panel, households, lookback, horizon, history_cols, future_cols, hid_to_idx, t0, t1
) -> ForecastDataset | None:
    try:
        batch = build_windows_in_span(
            panel, households, lookback, horizon, history_cols, future_cols, t0, t1
        )
    except RuntimeError:
        return None
    if batch.x.shape[0] == 0:
        return None
    return ForecastDataset(batch, hid_to_idx)


def _pretrain_global(
    panel: pd.DataFrame,
    split: TimeSplit,
    households: list[int],
    history_cols: list[str],
    future_cols: list[str],
    lookback: int,
    horizon: int,
    cfg: dict,
    seed: int,
) -> tuple[LSTMForecaster, object, dict]:
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
    return model, scaler, {"scaled": scaled, "hid_to_idx": hid_to_idx, "insample": insample, **info}


def _set_protocol_requires_grad(model: LSTMForecaster, protocol: str) -> list[torch.nn.Parameter]:
    if protocol == "P0":
        for p in model.parameters():
            p.requires_grad = False
        return []
    if protocol == "P1":
        for p in model.encoder_parameters():
            p.requires_grad = False
        params = list(model.head_parameters())
        for p in params:
            p.requires_grad = True
        return params
    if protocol == "P2":
        params = list(model.parameters())
        for p in params:
            p.requires_grad = True
        return params
    raise ValueError(protocol)


def _fine_tune_house(
    base: LSTMForecaster,
    protocol: str,
    train_ds: ForecastDataset | None,
    es_ds: ForecastDataset | None,
    scaler,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    seed: int,
    y_insample: np.ndarray,
) -> LSTMForecaster:
    model = copy.deepcopy(base).to(_device())
    params = _set_protocol_requires_grad(model, protocol)
    if protocol == "P0" or train_ds is None or len(train_ds) == 0:
        return model
    if not params:
        return model

    opt = torch.optim.Adam(params, lr=lr)
    loader = make_loader(train_ds, min(batch_size, len(train_ds)), shuffle=True, seed=seed)
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    bad = 0
    patience = max(3, max_epochs // 4)
    use_es = es_ds is not None and len(es_ds) > 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, fut, y, mask, hid in loader:
            device = next(model.parameters()).device
            x, fut, y, mask, hid = x.to(device), fut.to(device), y.to(device), mask.to(device), hid.to(device)
            pred = model(x, fut, hid)
            loss = masked_mse(pred, y, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if use_es:
            val = evaluate_kwh(model, es_ds, scaler, batch_size, y_insample=y_insample)
            score = val["mae"]
        else:
            # K too small for a held-out adapt slice: fixed budget, keep last epoch.
            score = -float(epoch)
        if score < best_mae:
            best_mae = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if use_es and bad >= patience:
            break
    model.load_state_dict(best_state)
    return model


def run_personalize(panel: pd.DataFrame, split: TimeSplit, exp: dict, data_cfg: dict, seed: int) -> list[dict]:
    """Return one metrics dict per (protocol, K) for logging."""
    weather_cfg = load_weather_config()
    weather_set = exp.get("weather_set") or "selected"
    if weather_set not in weather_cfg["sets"] and weather_set not in {"none", "temperature", "selected"}:
        raise ValueError(weather_set)

    pers = exp.get("personalization") or {}
    protocols = list(pers.get("protocols") or ["P0", "P1", "P2"])
    k_list = list(pers.get("k_days") or [1, 7, 30, 90])
    max_adapt_epochs = int(pers.get("max_epochs") or 20)
    lr_scale = float(pers.get("lr_scale") or 0.1)

    history_cols, future_cols = resolve_columns(weather_set, panel.columns)
    lookback = int(data_cfg["task"]["lookback_hours"])
    horizon = int(data_cfg["task"]["horizon_hours"])
    cfg = _model_cfg(exp)
    households = sorted(int(x) for x in panel["household_id"].unique())
    pretrain_lr = float(cfg.get("lr", 1e-3))
    adapt_lr = pretrain_lr * lr_scale
    batch_size = int(cfg.get("batch_size", 64))

    print(f"E08 pretrain global LSTM  seed={seed}", flush=True)
    set_seed(seed)
    t0 = time.perf_counter()
    base, scaler, ctx = _pretrain_global(
        panel, split, households, history_cols, future_cols, lookback, horizon, cfg, seed
    )
    scaled = ctx["scaled"]
    hid_to_idx = ctx["hid_to_idx"]
    insample = ctx["insample"]
    pretrain_s = float(ctx.get("train_seconds", 0.0))

    test_ds_all = ForecastDataset(
        build_windows(scaled, households, split, "test", lookback, horizon, history_cols, future_cols),
        hid_to_idx,
    )

    rows: list[dict] = []
    for protocol in protocols:
        for k_days in k_list:
            adapt_start, adapt_end = adaptation_window(split, int(k_days))
            a0, es0, a1 = _adapt_time_split(adapt_start, adapt_end)
            print(
                f"E08 {protocol} K={k_days}d  adapt={adapt_start}..{adapt_end}  es_from={es0}",
                flush=True,
            )
            preds, trues, masks = [], [], []
            per_house_mae: dict[int, float] = {}
            adapt_seconds = 0.0
            n_ok = 0
            for hid in households:
                t_house = time.perf_counter()
                train_ds = _try_span_dataset(
                    scaled, [hid], lookback, horizon, history_cols, future_cols, hid_to_idx, a0, es0
                )
                es_ds = _try_span_dataset(
                    scaled, [hid], lookback, horizon, history_cols, future_cols, hid_to_idx, es0, a1
                )
                # If early-stop slice empty, try using all adapt hours for train (K=1 case).
                if train_ds is None:
                    train_ds = _try_span_dataset(
                        scaled, [hid], lookback, horizon, history_cols, future_cols, hid_to_idx, a0, a1
                    )
                    es_ds = None

                model_h = _fine_tune_house(
                    base,
                    protocol,
                    train_ds,
                    es_ds,
                    scaler,
                    lr=adapt_lr,
                    batch_size=batch_size,
                    max_epochs=max_adapt_epochs if protocol != "P0" else 0,
                    seed=seed,
                    y_insample=insample,
                )
                adapt_seconds += time.perf_counter() - t_house

                # Test windows for this house only.
                try:
                    test_batch = build_windows(
                        scaled, [hid], split, "test", lookback, horizon, history_cols, future_cols
                    )
                except RuntimeError:
                    continue
                test_ds = ForecastDataset(test_batch, hid_to_idx)
                pred_s, true_s, mask, hids = predict_dataset(model_h, test_ds, batch_size)
                hid_h = np.repeat(hids, pred_s.shape[1])
                pred = scaler.inverse(pred_s.ravel(), hid_h).reshape(pred_s.shape)
                true = scaler.inverse(true_s.ravel(), hid_h).reshape(true_s.shape)
                preds.append(pred)
                trues.append(true)
                masks.append(mask)
                per_house_mae[hid] = mae_metric(true, pred, mask)
                n_ok += 1

            if not preds:
                raise RuntimeError(f"E08 {protocol} K={k_days}: no test predictions")
            metrics = masked_metrics(
                np.concatenate(trues),
                np.concatenate(preds),
                mask=np.concatenate(masks),
                y_insample=insample,
                seasonality=24,
            )
            metrics["train_seconds"] = pretrain_s + adapt_seconds
            metrics["n_params"] = sum(p.numel() for p in base.parameters())
            metrics["n_households"] = n_ok
            metrics["per_house_mae"] = per_house_mae
            metrics["notes"] = (
                f"E08 {protocol} K={k_days}d base=E04b; "
                f"houses={n_ok}; adapt_lr={adapt_lr}; "
                f"mean_house_mae={float(np.mean(list(per_house_mae.values()))):.4f}"
            )
            metrics["protocol"] = protocol
            metrics["k_days"] = int(k_days)
            print(
                f"E08 {protocol} K={k_days}  mae={metrics['mae']:.4f}  "
                f"wape={metrics['wape']:.3f}  houses={n_ok}",
                flush=True,
            )
            rows.append(metrics)

    # Reference: untouched global on full test (should match E04b ballpark).
    ref = evaluate_kwh(base, test_ds_all, scaler, batch_size, y_insample=insample)
    print(f"E08 pretrained global test MAE={ref['mae']:.4f} (reference)", flush=True)
    print(f"E08 seed={seed} wall={time.perf_counter() - t0:.0f}s", flush=True)
    return rows
