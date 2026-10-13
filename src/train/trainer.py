"""Train loop with early stopping on masked val MAE in kWh (not scaled units)."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ForecastDataset
from src.data.scalers import HouseLoadScaler
from src.eval.metrics import masked_metrics, per_household_mae_arrays


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask
    denom = weight.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * weight).sum() / denom


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset: ForecastDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
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
def predict_dataset(model: torch.nn.Module, dataset: ForecastDataset, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    preds, trues, masks, hids = [], [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    offset = 0
    for x, fut, y, mask, hid in loader:
        out = model(x.to(device), fut.to(device), hid.to(device)).cpu().numpy()
        n = out.shape[0]
        preds.append(out)
        trues.append(y.numpy())
        masks.append(mask.numpy() > 0.5)
        hids.append(dataset.household_id[offset : offset + n])
        offset += n
    return (
        np.concatenate(preds),
        np.concatenate(trues),
        np.concatenate(masks),
        np.concatenate(hids),
    )


def evaluate_kwh(
    model: torch.nn.Module,
    dataset: ForecastDataset,
    scaler: HouseLoadScaler,
    batch_size: int,
    y_insample: np.ndarray | None = None,
) -> dict[str, float]:
    pred_s, true_s, mask, hids = predict_dataset(model, dataset, batch_size)
    # Inverse per row: household_id is [N], y is [N, H]
    hid_h = np.repeat(hids, pred_s.shape[1])
    pred = scaler.inverse(pred_s.ravel(), hid_h).reshape(pred_s.shape)
    true = scaler.inverse(true_s.ravel(), hid_h).reshape(true_s.shape)
    metrics = masked_metrics(true, pred, mask=mask, y_insample=y_insample, seasonality=24)
    metrics["per_house_mae"] = per_household_mae_arrays(true, pred, hids, mask)
    return metrics


def train_model(
    model: torch.nn.Module,
    train_set: ForecastDataset,
    val_set: ForecastDataset,
    scaler: HouseLoadScaler,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    y_insample: np.ndarray | None = None,
) -> tuple[torch.nn.Module, dict]:
    device = _device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = make_loader(train_set, batch_size, shuffle=True, seed=seed)

    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    bad = 0
    t0 = time.perf_counter()
    n_train = len(train_set)
    print(
        f"train start  device={device}  train_windows={n_train}  "
        f"max_epochs={max_epochs}  patience={patience}",
        flush=True,
    )
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, fut, y, mask, hid in train_loader:
            x, fut, y, mask, hid = x.to(device), fut.to(device), y.to(device), mask.to(device), hid.to(device)
            pred = model(x, fut, hid)
            loss = masked_mse(pred, y, mask)
            opt.zero_grad()
            loss.backward()
            nn_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            del nn_clip
            opt.step()
        val = evaluate_kwh(model, val_set, scaler, batch_size, y_insample=y_insample)
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
