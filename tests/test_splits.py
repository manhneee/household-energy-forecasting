import pandas as pd

from src.data.splits import adaptation_window, chronological_split
from src.data.windowing import drop_crossing_windows


def test_split_is_chronological_and_contiguous():
    ts = pd.date_range("2014-01-01", periods=100, freq="h", tz="UTC")
    split = chronological_split(ts, 0.70, 0.15, 0.15)
    assert split.start == ts[0]
    assert split.train_end < split.val_end < split.test_end
    assert split.test_end == ts[-1]


def test_adaptation_never_overlaps_test():
    ts = pd.date_range("2014-01-01", periods=240, freq="h", tz="UTC")
    split = chronological_split(ts)
    start, end = adaptation_window(split, k_days=2)
    assert end <= split.val_end
    assert start < end
    assert end < split.val_end + pd.Timedelta(hours=1)


def test_test_windows_do_not_cross_into_val():
    ts = pd.date_range("2014-01-01", periods=240, freq="h", tz="UTC")
    split = chronological_split(ts)
    origins = drop_crossing_windows(ts, lookback=24, horizon=24, split=split, which="test")
    assert origins.min() > split.val_end
    last_target = origins.max() + pd.Timedelta(hours=23)
    assert last_target <= split.test_end
