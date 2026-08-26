import numpy as np
import pandas as pd

from src.data.scalers import HouseLoadScaler


def test_load_scaler_ignores_test_rows():
    ts = pd.date_range("2014-01-01", periods=10, freq="h", tz="UTC")
    panel = pd.DataFrame(
        {
            "household_id": [1] * 10,
            "timestamp": ts,
            "kwh": [1.0] * 7 + [100.0] * 3,
            "observed": [True] * 10,
            "split": ["train"] * 7 + ["test"] * 3,
        }
    )
    scaler = HouseLoadScaler().fit(panel)
    assert abs(scaler.mean_[1] - 1.0) < 1e-9
    scaled = scaler.transform_column(panel, "kwh")
    assert abs(scaled.iloc[0]) < 1e-9
