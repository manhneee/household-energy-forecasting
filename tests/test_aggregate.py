import pandas as pd

from src.data.aggregate import watts_to_hourly_kwh


def test_mean_watts_over_1000_is_kwh():
    start = pd.Timestamp("2014-01-15 10:00:00")
    rows = []
    for s in range(100):
        rows.append(
            {
                "household_id": 1,
                "datetime": start + pd.Timedelta(seconds=8 * s),
                "watts": 2000.0,
            }
        )
    hourly = watts_to_hourly_kwh(pd.DataFrame(rows), min_samples_per_hour=90)
    assert hourly["observed"].all()
    assert abs(hourly["kwh"].iloc[0] - 2.0) < 1e-9


def test_sparse_hour_is_missing():
    start = pd.Timestamp("2014-01-15 10:00:00")
    rows = [
        {"household_id": 1, "datetime": start + pd.Timedelta(seconds=8 * s), "watts": 800.0}
        for s in range(10)
    ]
    hourly = watts_to_hourly_kwh(pd.DataFrame(rows), min_samples_per_hour=90)
    assert not hourly["observed"].iloc[0]
    assert pd.isna(hourly["kwh"].iloc[0])


def test_exact_zero_hour_is_missing():
    start = pd.Timestamp("2014-01-15 10:00:00")
    rows = [
        {"household_id": 1, "datetime": start + pd.Timedelta(seconds=8 * s), "watts": 0.0}
        for s in range(120)
    ]
    hourly = watts_to_hourly_kwh(pd.DataFrame(rows), min_samples_per_hour=90)
    assert not hourly["observed"].iloc[0]
