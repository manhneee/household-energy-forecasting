"""Synthetic neighborhood panel so LSTM can be smoke-tested without the 2 GB REFIT download."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_data_config
from src.data.features import add_calendar_features
from src.data.splits import assign_split, chronological_split
from src.paths import processed_dir


def make_demo_panel(
    n_households: int = 6,
    n_days: int = 180,
    start: str = "2014-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=n_days * 24, freq="h", tz="UTC")
    local_hour = hours.tz_convert("Europe/London").hour
    # Shared town weather: cold winters, solar peak near local noon.
    day_of_year = hours.dayofyear.to_numpy()
    temperature = 9 - 7 * np.cos(2 * np.pi * (day_of_year - 15) / 365) + rng.normal(0, 1.2, len(hours))
    radiation = np.clip(800 * np.sin(np.pi * (local_hour.to_numpy() - 6) / 12), 0, None)
    radiation = radiation * (0.4 + 0.6 * (1 - np.cos(2 * np.pi * day_of_year / 365) / 2))
    humidity = np.clip(80 - 0.8 * temperature + rng.normal(0, 4, len(hours)), 40, 100)

    frames = []
    for i, hid in enumerate(range(1, n_households + 1)):
        base = 0.35 + 0.12 * i
        daily = 0.25 * np.sin(2 * np.pi * (local_hour.to_numpy() - 7) / 24)
        weekly = 0.05 * ((hours.dayofweek.to_numpy() >= 5).astype(float))
        # Weak weather coupling: lighting / cold appliances, not electric heat.
        weather = -0.008 * (temperature - 10) + 0.00004 * radiation
        noise = rng.normal(0, 0.04, len(hours))
        kwh = np.clip(base + daily + weekly + weather + noise, 0.02, None)
        observed = rng.random(len(hours)) > 0.04
        kwh = np.where(observed, kwh, np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "household_id": hid,
                    "timestamp": hours,
                    "kwh": kwh,
                    "observed": observed,
                    "n_samples": np.where(observed, 400, 0).astype(int),
                    "temperature_2m": temperature,
                    "shortwave_radiation": radiation,
                    "relative_humidity_2m": humidity,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    cal = add_calendar_features(panel["timestamp"])
    panel = pd.concat([panel.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)
    # Short-gap input fill; mask stays False on those hours.
    panel["kwh_input"] = panel["kwh"]
    filled = []
    for _, g in panel.groupby("household_id"):
        g = g.sort_values("timestamp").copy()
        series = g.set_index("timestamp")["kwh"].astype(float)
        g["kwh_input"] = series.interpolate(method="time", limit=3, limit_direction="both").to_numpy()
        filled.append(g)
    panel = pd.concat(filled, ignore_index=True)
    split = chronological_split(pd.DatetimeIndex(hours))
    panel["split"] = assign_split(panel["timestamp"], split)
    return panel, split


def write_demo_panel() -> None:
    cfg = load_data_config()
    panel, split = make_demo_panel()
    out = processed_dir(cfg)
    panel.to_parquet(out / "hourly_panel.parquet", index=False)
    pd.DataFrame(
        [
            {
                "train_end": split.train_end,
                "val_end": split.val_end,
                "test_end": split.test_end,
                "start": split.start,
                "kept": sorted(panel["household_id"].unique().tolist()),
                "alignment": {"ok": True, "note": "synthetic demo panel"},
            }
        ]
    ).to_json(out / "split.json", orient="records", date_format="iso")
    print(f"Wrote demo panel to {out / 'hourly_panel.parquet'} shape={panel.shape}")
    print(f"split train_end={split.train_end} val_end={split.val_end} test_end={split.test_end}")
