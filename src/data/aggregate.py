"""Aggregate 8-second Watts to hourly kWh with a sample-count guard.

hourly_kwh = mean(watts) / 1000
That is average power in kW over a one-hour bin, which equals kWh for that hour.

Hours with too few 8-second samples, or exact-zero hours after the cleaned
release's zero-fill, are marked missing. They must not look like real load.
"""

from __future__ import annotations

import pandas as pd


def watts_to_hourly_kwh(
    raw: pd.DataFrame,
    min_samples_per_hour: int = 90,
    treat_zero_hours_as_missing: bool = True,
) -> pd.DataFrame:
    if raw.empty:
        return _empty_hourly()

    work = raw.dropna(subset=["watts"]).copy() if "watts" in raw.columns else raw.copy()
    # Prefer Unix (UTC). Wall-clock DateTime hits AmbiguousTimeError on the
    # October DST fallback; the paper's Unix column is already UTC.
    if "unix" in work.columns and work["unix"].notna().any():
        work["utc"] = pd.to_datetime(work["unix"], unit="s", utc=True)
    else:
        work = work.dropna(subset=["datetime"]).copy()
        if work["datetime"].dt.tz is None:
            work["datetime"] = work["datetime"].dt.tz_localize(
                "Europe/London",
                ambiguous="NaT",
                nonexistent="NaT",
            )
        work = work.dropna(subset=["datetime"])
        work["utc"] = work["datetime"].dt.tz_convert("UTC")
    work["hour"] = work["utc"].dt.floor("h")

    grouped = work.groupby(["household_id", "hour"], sort=True)
    hourly = grouped.agg(
        watts_mean=("watts", "mean"),
        n_samples=("watts", "count"),
        n_zero=("watts", lambda s: int((s == 0).sum())),
    ).reset_index()

    observed = hourly["n_samples"] >= min_samples_per_hour
    if treat_zero_hours_as_missing:
        # A real vacant house still has a fridge. Exact 0 W across a well-sampled
        # hour is the cleaned-release gap fill (gaps > 2 min become zeros).
        observed &= hourly["watts_mean"].fillna(0).ne(0)

    hourly["kwh"] = hourly["watts_mean"] / 1000.0
    hourly.loc[~observed, "kwh"] = pd.NA
    hourly["observed"] = observed
    hourly = hourly.rename(columns={"hour": "timestamp"})
    return hourly[["household_id", "timestamp", "kwh", "observed", "n_samples"]]


def _empty_hourly() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["household_id", "timestamp", "kwh", "observed", "n_samples"]
    )
