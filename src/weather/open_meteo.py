"""Cached Open-Meteo Historical Weather (ERA5) for Loughborough.

Timezone rule: request UTC. REFIT timestamps are already UTC after
src.data.aggregate.watts_to_hourly_kwh. Merge on the UTC hour.

Future weather on the 24-hour horizon is this same ERA5 series (oracle).
Do not pretend these are 2013-2015 operational forecasts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.paths import weather_dir


def fetch_era5(
    start: pd.Timestamp,
    end: pd.Timestamp,
    weather_cfg: dict,
    data_cfg: dict,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    om = weather_cfg["open_meteo"]
    cache = weather_dir(data_cfg) / om["cache_name"]
    meta_path = weather_dir(data_cfg) / om["metadata_name"]

    start_d = pd.Timestamp(start).tz_convert("UTC").strftime("%Y-%m-%d")
    end_d = pd.Timestamp(end).tz_convert("UTC").strftime("%Y-%m-%d")

    if cache.exists():
        cached = pd.read_parquet(cache)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        if cached["timestamp"].min() <= pd.Timestamp(start_d, tz="UTC") and cached[
            "timestamp"
        ].max() >= pd.Timestamp(end_d, tz="UTC"):
            return cached

    params = {
        "latitude": om["latitude"],
        "longitude": om["longitude"],
        "start_date": start_d,
        "end_date": end_d,
        "hourly": ",".join(om["hourly"]),
        "timezone": om["timezone"],
    }
    sess = session or requests.Session()
    resp = sess.get(om["endpoint"], params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload["hourly"]
    ts = pd.to_datetime(hourly["time"], utc=True)
    frame = pd.DataFrame({"timestamp": ts})
    for col in om["hourly"]:
        frame[col] = hourly[col]
    frame.to_parquet(cache, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "endpoint": om["endpoint"],
                "params": params,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "source": payload.get("hourly_units", {}),
                "note": "ERA5 reanalysis via Open-Meteo Historical Weather API. Horizon weather is oracle.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame


def merge_weather(panel: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)
    out = panel.merge(w, on="timestamp", how="left")
    return out


def verify_solar_noon(
    weather: pd.DataFrame,
    local_tz: str = "Europe/London",
    expected_hour: int = 12,
    tolerance_hours: int = 2,
) -> dict:
    """The daily shortwave peak must sit near local noon. A 1-hour DST shift is a silent bug."""
    if "shortwave_radiation" not in weather.columns:
        return {"ok": False, "reason": "shortwave_radiation missing"}
    w = weather.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)
    local = w["timestamp"].dt.tz_convert(local_tz)
    w["local_hour"] = local.dt.hour
    w["local_date"] = local.dt.date
    peaks = w.loc[w.groupby("local_date")["shortwave_radiation"].idxmax()]
    mean_hour = float(peaks["local_hour"].mean())
    ok = abs(mean_hour - expected_hour) <= tolerance_hours
    return {
        "ok": ok,
        "mean_local_peak_hour": mean_hour,
        "expected_hour": expected_hour,
        "tolerance_hours": tolerance_hours,
        "n_days": int(peaks.shape[0]),
    }
