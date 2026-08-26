from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_house(path: Path, household_id: int, start: str, hours: int, watt_fn, drop_frac: float = 0.0) -> None:
    rng = np.random.default_rng(household_id)
    start_ts = pd.Timestamp(start)
    rows = []
    for h in range(hours):
        hour = start_ts + pd.Timedelta(hours=h)
        # ~120 8-second-like samples per hour (enough to pass min_samples=90)
        n = 120
        if drop_frac and rng.random() < drop_frac:
            continue
        watts = watt_fn(h)
        for s in range(n):
            t = hour + pd.Timedelta(seconds=8 * s)
            rows.append((t.strftime("%Y-%m-%d %H:%M:%S"), watts + rng.normal(0, 1)))
    df = pd.DataFrame(rows, columns=["DateTime", "Aggregate"])
    df.to_csv(path / f"CLEAN_House{household_id}.csv", index=False)


@pytest.fixture
def tiny_refit(tmp_path: Path) -> Path:
    """Four houses, winter 2014, no DST. House 3 is solar-contaminated (still written)."""

    def daily(base, h, phase):
        return base + 80 * np.sin(2 * np.pi * ((h + phase) % 24) / 24)

    _write_house(tmp_path, 1, "2014-01-01", 14 * 24, lambda h: daily(300, h, 0))
    _write_house(tmp_path, 2, "2014-01-01", 14 * 24, lambda h: daily(450, h, 6), drop_frac=0.05)
    _write_house(tmp_path, 3, "2014-01-01", 14 * 24, lambda h: daily(200, h, 3))
    _write_house(tmp_path, 5, "2014-01-01", 14 * 24, lambda h: daily(350, h, 12))
    return tmp_path
