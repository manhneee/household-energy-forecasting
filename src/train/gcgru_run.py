"""E06 / E07: GCGRU on train-only Pearson kNN graph."""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_weather_config
from src.data.feature_sets import resolve_columns
from src.data.graph_dataset import GraphForecastDataset, build_graph_windows
from src.data.splits import TimeSplit
from src.eval.metrics import mae as mae_metric
from src.eval.metrics import masked_metrics
from src.graphs.correlation import pearson_knn
from src.models.gcgru import GCGRUForecaster
from src.train.lstm_run import _insample_kwh, _model_cfg, _scale_panel
from src.train.trainer import masked_mse


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(dataset: GraphForecastDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=0,
    )


@torch.no_grad()
def _predict(model: GCGRUForecaster, dataset: GraphForecastDataset, batch_size: int):
    model.eval()
    device = next(model.parameters()).device
    preds, trues, masks = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for x, fut, y, mask in loader:
        out = model(x.to(device), fut.to(device)).cpu().numpy()
        preds.append(out)
        trues.append(y.numpy())
        masks.append(mask.numpy() > 0.5)
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)


def _evaluate_kwh(
    model: GCGRUForecaster,
    dataset: GraphForecastDataset,
    scaler,
    households: list[int],
    batch_size: int,
    y_insample: np.ndarray | None,
) -> dict[str, float]:
    pred_s, true_s, mask = _predict(model, dataset, batch_size)
    # pred_s: [S, N, H]
    s, n, h = pred_s.shape
    pred = np.empty_like(pred_s)
    true = np.empty_like(true_s)
    for i, hid in enumerate(households):
        ids = np.full(s * h, hid, dtype=np.int64)
        pred[:, i, :] = scaler.inverse(pred_s[:, i, :].ravel(), ids).reshape(s, h)
        true[:, i, :] = scaler.inverse(true_s[:, i, :].ravel(), ids).reshape(s, h)
    metrics = masked_metrics(true, pred, mask=mask, y_insample=y_insample, seasonality=24)
    per_house: dict[int, float] = {}
    for i, hid in enumerate(households):
        per_house[int(hid)] = mae_metric(true[:, i, :], pred[:, i, :], mask[:, i, :])
    metrics["per_house_mae"] = per_house
    return metrics


def _train_graph(
    model: GCGRUForecaster,
    train_set: GraphForecastDataset,
    val_set: GraphForecastDataset,
    scaler,
    households: list[int],
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    y_insample: np.ndarray | None,
) -> tuple[GCGRUForecaster, dict]:
    device = _device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _make_loader(train_set, batch_size, shuffle=True, seed=seed)
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    bad = 0
    t0 = time.perf_counter()
    print(
        f"train start  device={device}  train_windows={len(train_set)}  "
        f"N={len(households)}  max_epochs={max_epochs}  patience={patience}",
        flush=True,
    )
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, fut, y, mask in train_loader:
            x, fut, y, mask = x.to(device), fut.to(device), y.to(device), mask.to(device)
            pred = model(x, fut)
            loss = masked_mse(pred, y, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = _evaluate_kwh(model, val_set, scaler, households, batch_size, y_insample)
        if val["mae"] < best_mae:
            best_mae = val["mae"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            tag = "best"
        else:
            bad += 1
            tag = f"wait {bad}/{patience}"
        elapsed = time.perf_counter() - t0
        print(
            f"epoch {epoch}/{max_epochs}  val_mae={val['mae']:.4f} kWh  "
            f"best={best_mae:.4f}  {tag}  {elapsed:.0f}s",
            flush=True,
        )
        if bad >= patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    print(f"train done  best_val_mae={best_mae:.4f} kWh  {elapsed:.0f}s", flush=True)
    return model, {"train_seconds": elapsed, "best_val_mae": best_mae, "epochs": epoch}


def run_gcgru(panel: pd.DataFrame, split: TimeSplit, exp: dict, data_cfg: dict, seed: int) -> dict:
    weather_cfg = load_weather_config()
    weather_set = exp.get("weather_set") or "none"
    if weather_set not in weather_cfg["sets"] and weather_set not in {"none", "temperature", "selected"}:
        raise ValueError(weather_set)
    if exp.get("mode", "graph") != "graph":
        raise NotImplementedError("E06/E07 require mode=graph")

    history_cols, future_cols = resolve_columns(weather_set, panel.columns)
    lookback = int(data_cfg["task"]["lookback_hours"])
    horizon = int(data_cfg["task"]["horizon_hours"])
    cfg = _model_cfg(exp)
    graph_cfg = cfg.get("graph") or {}
    k = int(graph_cfg.get("k", 3))
    households = sorted(int(x) for x in panel["household_id"].unique())

    scaled, scaler = _scale_panel(panel, history_cols, future_cols)
    adj, _corr = pearson_knn(scaled, households, k=k)
    print(f"Pearson kNN k={k}  adj mean={float(adj.mean()):.4f}  shape={adj.shape}", flush=True)

    train_ds = GraphForecastDataset(
        build_graph_windows(scaled, households, split, "train", lookback, horizon, history_cols, future_cols)
    )
    val_ds = GraphForecastDataset(
        build_graph_windows(scaled, households, split, "val", lookback, horizon, history_cols, future_cols)
    )
    test_ds = GraphForecastDataset(
        build_graph_windows(scaled, households, split, "test", lookback, horizon, history_cols, future_cols)
    )

    model = GCGRUForecaster(
        n_history=len(history_cols),
        n_future=len(future_cols),
        horizon=horizon,
        n_households=len(households),
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.set_adjacency(adj)
    insample = _insample_kwh(panel)
    model, train_info = _train_graph(
        model,
        train_ds,
        val_ds,
        scaler,
        households,
        lr=float(cfg.get("lr", 1e-3)),
        batch_size=int(cfg.get("batch_size", 16)),
        max_epochs=int(cfg.get("max_epochs", 40)),
        patience=int(cfg.get("patience", 8)),
        seed=seed,
        y_insample=insample,
    )
    metrics = _evaluate_kwh(
        model, test_ds, scaler, households, int(cfg.get("batch_size", 16)), insample
    )
    metrics.update(train_info)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    metrics["n_households"] = len(households)
    return metrics
