"""Calendar features. UK bank holidays for England & Wales, 2013-2015, hardcoded.

A holidays package would pull a network dependency into every Kaggle run for
eight dates per year. The list is small and must stay frozen for reproducibility.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# England and Wales bank holidays covering the REFIT span.
UK_BANK_HOLIDAYS = {
    date(2013, 1, 1),
    date(2013, 3, 29),
    date(2013, 4, 1),
    date(2013, 5, 6),
    date(2013, 5, 27),
    date(2013, 8, 26),
    date(2013, 12, 25),
    date(2013, 12, 26),
    date(2014, 1, 1),
    date(2014, 4, 18),
    date(2014, 4, 21),
    date(2014, 5, 5),
    date(2014, 5, 26),
    date(2014, 8, 25),
    date(2014, 12, 25),
    date(2014, 12, 26),
    date(2015, 1, 1),
    date(2015, 4, 3),
    date(2015, 4, 6),
    date(2015, 5, 4),
    date(2015, 5, 25),
    date(2015, 8, 31),
    date(2015, 12, 25),
    date(2015, 12, 28),  # Boxing Day substitute
}


def add_calendar_features(timestamps: pd.DatetimeIndex | pd.Series, local_tz: str = "Europe/London") -> pd.DataFrame:
    utc = pd.DatetimeIndex(timestamps)
    if utc.tz is None:
        utc = utc.tz_localize("UTC")
    else:
        utc = utc.tz_convert("UTC")
    local = utc.tz_convert(local_tz)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    hol = np.array([d.date() in UK_BANK_HOLIDAYS for d in local], dtype=np.float32)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_weekend": (dow >= 5).astype(np.float32),
            "is_uk_bank_holiday": hol,
        },
        index=utc,
    )
