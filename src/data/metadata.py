"""Household characteristics from Murray et al. (2017), Scientific Data 4:160122, Table 2.

The Kaggle upload is not guaranteed to include this table. Values are transcribed
from the paper and cited as such. House 14 does not exist.
"""

from __future__ import annotations

import math
import re

import pandas as pd

# house_id: occupancy, dwelling_age (as published), n_appliances, dwelling_type, size
TABLE2: dict[int, dict] = {
    1: {"occupancy": 2, "dwelling_age": "1975-1980", "n_appliances": 35, "dwelling_type": "Detached", "size": "4 bed"},
    2: {"occupancy": 4, "dwelling_age": None, "n_appliances": 15, "dwelling_type": "Semi-detached", "size": "3 bed"},
    3: {"occupancy": 2, "dwelling_age": "1988", "n_appliances": 27, "dwelling_type": "Detached", "size": "3 bed"},
    4: {"occupancy": 2, "dwelling_age": "1850-1899", "n_appliances": 33, "dwelling_type": "Detached", "size": "4 bed"},
    5: {"occupancy": 4, "dwelling_age": "1878", "n_appliances": 44, "dwelling_type": "Mid-terrace", "size": "4 bed"},
    6: {"occupancy": 2, "dwelling_age": "2005", "n_appliances": 49, "dwelling_type": "Detached", "size": "4 bed"},
    7: {"occupancy": 4, "dwelling_age": "1965-1974", "n_appliances": 25, "dwelling_type": "Detached", "size": "3 bed"},
    8: {"occupancy": 2, "dwelling_age": "1966", "n_appliances": 35, "dwelling_type": "Detached", "size": "2 bed"},
    9: {"occupancy": 2, "dwelling_age": "1919-1944", "n_appliances": 24, "dwelling_type": "Detached", "size": "3 bed"},
    10: {"occupancy": 4, "dwelling_age": "1919-1944", "n_appliances": 31, "dwelling_type": "Detached", "size": "3 bed"},
    11: {"occupancy": 1, "dwelling_age": "1945-1964", "n_appliances": 25, "dwelling_type": "Detached", "size": "3 bed"},
    12: {"occupancy": 3, "dwelling_age": "1991-1995", "n_appliances": 26, "dwelling_type": "Detached", "size": "3 bed"},
    13: {"occupancy": 4, "dwelling_age": "post 2002", "n_appliances": 28, "dwelling_type": "Detached", "size": "4 bed"},
    15: {"occupancy": 1, "dwelling_age": "1965-1974", "n_appliances": 19, "dwelling_type": "Semi-detached", "size": "3 bed"},
    16: {"occupancy": 6, "dwelling_age": "1981-1990", "n_appliances": 48, "dwelling_type": "Detached", "size": "5 bed"},
    17: {"occupancy": 3, "dwelling_age": "mid 60s", "n_appliances": 22, "dwelling_type": "Detached", "size": "3 bed"},
    18: {"occupancy": 2, "dwelling_age": "1965-1974", "n_appliances": 34, "dwelling_type": "Detached", "size": "3 bed"},
    19: {"occupancy": 4, "dwelling_age": "1945-1964", "n_appliances": 26, "dwelling_type": "Semi-detached", "size": "3 bed"},
    20: {"occupancy": 2, "dwelling_age": "1965-1974", "n_appliances": 39, "dwelling_type": "Detached", "size": "3 bed"},
    21: {"occupancy": 4, "dwelling_age": "1981-1990", "n_appliances": 23, "dwelling_type": "Detached", "size": "3 bed"},
}

SOLAR_CONTAMINATED = (3, 11, 21)
REWIRED_SOLAR = (1, 6, 7)
CITATION = "Murray, Stankovic, and Stankovic (2017), Scientific Data 4:160122, Table 2"


def _age_year(label: str | None) -> float:
    if label is None:
        return math.nan
    text = str(label).strip().lower()
    if text in {"", "-", "—", "none", "nan"}:
        return math.nan
    if "mid 60" in text:
        return 1965.0
    if "post 2002" in text:
        return 2003.0
    years = [int(x) for x in re.findall(r"\d{4}", text)]
    if not years:
        return math.nan
    return float(sum(years) / len(years))


def _bedrooms(size: str) -> int:
    match = re.search(r"(\d+)", size)
    if not match:
        raise ValueError(f"Cannot parse bedroom count from {size!r}")
    return int(match.group(1))


def household_table() -> pd.DataFrame:
    rows = []
    for house_id, row in TABLE2.items():
        rows.append(
            {
                "household_id": house_id,
                "occupancy": row["occupancy"],
                "dwelling_age": row["dwelling_age"],
                "dwelling_year": _age_year(row["dwelling_age"]),
                "n_appliances": row["n_appliances"],
                "dwelling_type": row["dwelling_type"],
                "bedrooms": _bedrooms(row["size"]),
                "solar_contaminated": house_id in SOLAR_CONTAMINATED,
                "rewired_pv": house_id in REWIRED_SOLAR,
                "source": CITATION,
            }
        )
    return pd.DataFrame(rows).sort_values("household_id").reset_index(drop=True)


def numeric_features(household_ids: list[int]) -> pd.DataFrame:
    """Static node features for the metadata graph / embeddings."""
    table = household_table().set_index("household_id")
    type_dummies = pd.get_dummies(table["dwelling_type"], prefix="type")
    feats = pd.concat(
        [table[["occupancy", "dwelling_year", "n_appliances", "bedrooms"]], type_dummies],
        axis=1,
    )
    return feats.reindex(household_ids)
