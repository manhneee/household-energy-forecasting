"""End-to-end panel build: REFIT -> hourly kWh -> quality -> weather -> split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import load_data_config, load_weather_config
from src.data.aggregate import watts_to_hourly_kwh
from src.data.features import add_calendar_features
from src.data.loading import discover_house_files, iter_raw_houses
from src.data.quality import (
    choose_common_window,
    dense_panel,
    impute_short_gaps,
    per_house_span,
    write_quality_report,
)
from src.data.splits import TimeSplit, assign_split, chronological_split
from src.paths import figures_dir, processed_dir, reports_dir, resolve_raw_dir
from src.weather.open_meteo import fetch_era5, merge_weather, verify_solar_noon


@dataclass
class PanelBundle:
    panel: pd.DataFrame
    split: TimeSplit
    kept: list[int]
    alignment: dict
    raw_dir: Path


def build_hourly_panel(
    data_cfg: dict | None = None,
    weather_cfg: dict | None = None,
    include_solar: bool = False,
    fetch_weather: bool = True,
) -> PanelBundle:
    data_cfg = data_cfg or load_data_config()
    weather_cfg = weather_cfg or load_weather_config()
    raw_dir = resolve_raw_dir(data_cfg)
    solar = [] if include_solar else list(data_cfg["exclusions"]["solar_contaminated"])

    discovered = discover_house_files(raw_dir)
    print(f"Discovered {len(discovered)} REFIT files in {raw_dir}: {sorted(discovered)}")
    if 14 in discovered:
        raise RuntimeError("House 14 should not exist")

    hourly_parts = []
    for _, raw in iter_raw_houses(raw_dir, exclude=solar):
        hourly_parts.append(
            watts_to_hourly_kwh(
                raw,
                min_samples_per_hour=data_cfg["aggregation"]["min_samples_per_hour"],
                treat_zero_hours_as_missing=data_cfg["aggregation"]["treat_zero_hours_as_missing"],
            )
        )
        del raw
    hourly = pd.concat(hourly_parts, ignore_index=True)
    spans = per_house_span(hourly)
    clipped, decision = choose_common_window(
        hourly,
        coverage_threshold=data_cfg["quality"]["coverage_threshold"],
        min_common_months=data_cfg["quality"]["min_common_months"],
    )
    write_quality_report(
        reports_dir(data_cfg) / "data_quality.md",
        decision,
        spans,
        hourly,
        solar,
    )
    if not decision.kept:
        raise RuntimeError("No households survived the quality protocol. See reports/data_quality.md")

    panel = dense_panel(clipped, decision.window_start, decision.window_end)
    panel = impute_short_gaps(panel, max_impute_hours=data_cfg["quality"]["max_impute_hours"])

    alignment = {}
    if fetch_weather:
        weather = fetch_era5(decision.window_start, decision.window_end, weather_cfg, data_cfg)
        alignment = verify_solar_noon(
            weather,
            local_tz=weather_cfg["open_meteo"]["local_timezone"],
            expected_hour=weather_cfg["alignment_check"]["expected_local_hour"],
            tolerance_hours=weather_cfg["alignment_check"]["tolerance_hours"],
        )
        if not alignment["ok"]:
            raise RuntimeError(f"Weather UTC/DST alignment failed: {alignment}")
        panel = merge_weather(panel, weather)

    cal = add_calendar_features(panel["timestamp"], weather_cfg["open_meteo"]["local_timezone"])
    cal = cal.reset_index(drop=True)
    panel = pd.concat([panel.reset_index(drop=True), cal], axis=1)

    timestamps = pd.DatetimeIndex(sorted(panel["timestamp"].unique()))
    split = chronological_split(
        timestamps,
        train_frac=data_cfg["split"]["train"],
        val_frac=data_cfg["split"]["val"],
        test_frac=data_cfg["split"]["test"],
    )
    panel["split"] = assign_split(panel["timestamp"], split)

    out_dir = processed_dir(data_cfg)
    panel.to_parquet(out_dir / "hourly_panel.parquet", index=False)
    pd.DataFrame(
        [
            {
                "train_end": split.train_end,
                "val_end": split.val_end,
                "test_end": split.test_end,
                "start": split.start,
                "kept": decision.kept,
                "alignment": alignment,
            }
        ]
    ).to_json(out_dir / "split.json", orient="records", date_format="iso")

    figures_dir(data_cfg)  # ensure the folder exists
    print(f"Kept N={len(decision.kept)} houses {decision.kept}")
    print(f"Window {decision.window_start} -> {decision.window_end} ({decision.n_hours} h)")
    print(f"Split train_end={split.train_end} val_end={split.val_end} test_end={split.test_end}")
    print(f"Alignment {alignment}")
    return PanelBundle(panel=panel, split=split, kept=decision.kept, alignment=alignment, raw_dir=raw_dir)


def load_processed_panel(data_cfg: dict | None = None) -> tuple[pd.DataFrame, TimeSplit]:
    data_cfg = data_cfg or load_data_config()
    path = processed_dir(data_cfg) / "hourly_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/build_panel.py first.")
    panel = pd.read_parquet(path)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    meta = pd.read_json(processed_dir(data_cfg) / "split.json")
    def _ts(value) -> pd.Timestamp:
        t = pd.Timestamp(value)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    split = TimeSplit(
        start=_ts(meta.loc[0, "start"]),
        train_end=_ts(meta.loc[0, "train_end"]),
        val_end=_ts(meta.loc[0, "val_end"]),
        test_end=_ts(meta.loc[0, "test_end"]),
    )
    return panel, split
