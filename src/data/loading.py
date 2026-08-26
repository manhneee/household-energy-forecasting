"""Discover and read REFIT house CSVs.

The official cleaned release uses CLEAN_House1.csv. The Kaggle upload
kyleahmurphy/uk-electrical-load uses House_1.csv. Both are accepted.
House IDs run 1-21 with no House 14. Never generate IDs with range(1, 21).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

HOUSE_FILE_RE = re.compile(r"(?:CLEAN_House|House_)(\d+)\.csv$", re.IGNORECASE)


def discover_house_files(raw_dir: str | Path) -> dict[int, Path]:
    raw_dir = Path(raw_dir)
    found: dict[int, Path] = {}
    if not raw_dir.exists():
        return found
    candidates = list(raw_dir.glob("CLEAN_House*.csv")) + list(raw_dir.glob("House_*.csv"))
    for path in sorted(set(candidates)):
        match = HOUSE_FILE_RE.search(path.name)
        if not match:
            continue
        house_id = int(match.group(1))
        found[house_id] = path
    if 14 in found:
        raise RuntimeError(f"Found {found[14].name}; REFIT should not contain House 14")
    return found


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in {"datetime", "time", "timestamp"} and "unix" not in key:
            rename[col] = "datetime"
        elif key in {"unix", "unixtimestamp", "unix_timestamp"}:
            rename[col] = "unix"
        elif key in {"aggregate", "agg"}:
            rename[col] = "watts"
    df = df.rename(columns=rename)
    if "datetime" not in df.columns and "unix" not in df.columns:
        raise ValueError(f"No DateTime/Unix column in {list(df.columns)}")
    if "watts" not in df.columns:
        raise ValueError(f"No Aggregate column in {list(df.columns)}")
    return df


def _keep_raw_column(name: str) -> bool:
    key = str(name).strip().lower().replace(" ", "")
    return key in {
        "datetime",
        "time",
        "unix",
        "unixtimestamp",
        "unix_timestamp",
        "aggregate",
        "agg",
    }


def read_house(path: Path, household_id: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=_keep_raw_column)
    df = _normalize_columns(df)
    keep = ["watts"]
    if "datetime" in df.columns:
        keep.insert(0, "datetime")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "unix" in df.columns:
        keep.append("unix")
        df["unix"] = pd.to_numeric(df["unix"], errors="coerce")
    df = df[keep].copy()
    df["watts"] = pd.to_numeric(df["watts"], errors="coerce")
    if "datetime" in df.columns:
        df = df.dropna(subset=["datetime"])
    df["household_id"] = household_id
    return df


def iter_raw_houses(
    raw_dir: str | Path,
    exclude: list[int] | None = None,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield one house at a time so the 8-second files are not all in RAM."""
    files = discover_house_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No House_*.csv or CLEAN_House*.csv files under {raw_dir}. "
            "Download kyleahmurphy/uk-electrical-load or set REFIT_RAW_DIR."
        )
    exclude_set = set(exclude or [])
    for house_id, path in files.items():
        if house_id in exclude_set:
            continue
        print(f"Reading {path.name} (house {house_id})")
        yield house_id, read_house(path, house_id)


def load_raw_houses(
    raw_dir: str | Path,
    exclude: list[int] | None = None,
) -> dict[int, pd.DataFrame]:
    return {hid: df for hid, df in iter_raw_houses(raw_dir, exclude=exclude)}
