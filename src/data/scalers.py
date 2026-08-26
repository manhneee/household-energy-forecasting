"""Train-only scalers. Fit after the split is assigned, never on val/test."""

from __future__ import annotations

import numpy as np
import pandas as pd


class HouseLoadScaler:
    """Per-household mean/std of observed train kWh."""

    def __init__(self) -> None:
        self.mean_: dict[int, float] = {}
        self.std_: dict[int, float] = {}

    def fit(self, panel: pd.DataFrame) -> HouseLoadScaler:
        train = panel[panel["split"] == "train"]
        for hid, g in train.groupby("household_id"):
            x = pd.to_numeric(g.loc[g["observed"], "kwh"], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            mu = float(np.mean(x)) if x.size else 0.0
            sd = float(np.std(x)) if x.size else 1.0
            self.mean_[int(hid)] = mu
            self.std_[int(hid)] = sd if sd > 1e-6 else 1.0
        return self

    def transform_column(self, panel: pd.DataFrame, column: str) -> pd.Series:
        out = pd.to_numeric(panel[column], errors="coerce").astype(float)
        for hid, mu in self.mean_.items():
            sd = self.std_[hid]
            mask = panel["household_id"] == hid
            out.loc[mask] = (out.loc[mask] - mu) / sd
        return out

    def inverse(self, values: np.ndarray, household_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        household_ids = np.asarray(household_ids)
        out = values.copy()
        for hid, mu in self.mean_.items():
            sd = self.std_[hid]
            sel = household_ids == hid
            out[sel] = out[sel] * sd + mu
        return out


class ColumnScaler:
    """Shared scaler for weather. One series for the whole town, still train-only."""

    def __init__(self) -> None:
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, panel: pd.DataFrame, columns: list[str]) -> ColumnScaler:
        train = panel[panel["split"] == "train"]
        for col in columns:
            if col not in panel.columns:
                continue
            x = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            mu = float(np.mean(x)) if x.size else 0.0
            sd = float(np.std(x)) if x.size else 1.0
            self.mean_[col] = mu
            self.std_[col] = sd if sd > 1e-6 else 1.0
        return self

    def transform(self, panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = panel.copy()
        for col in columns:
            if col not in out.columns or col not in self.mean_:
                continue
            out[col] = (pd.to_numeric(out[col], errors="coerce") - self.mean_[col]) / self.std_[col]
        return out
