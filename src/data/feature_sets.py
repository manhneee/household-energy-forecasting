"""Which columns enter the model. Calendar is known future; weather on the horizon is ERA5 oracle."""

from __future__ import annotations

CALENDAR_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_uk_bank_holiday",
]

WEATHER_SETS = {
    "none": [],
    "temperature": ["temperature_2m"],
    "selected": ["temperature_2m", "shortwave_radiation", "relative_humidity_2m"],
}


def resolve_columns(weather_set: str, available: list[str] | set[str]) -> tuple[list[str], list[str]]:
    """Return (history_cols, future_cols). History includes load; future does not."""
    if weather_set not in WEATHER_SETS:
        raise ValueError(f"Unknown weather_set {weather_set!r}. Use none / temperature / selected.")
    available = set(available)
    weather = [c for c in WEATHER_SETS[weather_set] if c in available]
    calendar = [c for c in CALENDAR_COLS if c in available]
    history = ["kwh_input", *calendar, *weather]
    future = [*calendar, *weather]
    return history, future
