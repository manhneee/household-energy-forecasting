import numpy as np
import pandas as pd

from src.data.aggregate import watts_to_hourly_kwh
from src.data.loading import load_raw_houses
from src.data.quality import choose_common_window, dense_panel, impute_short_gaps
from src.data.splits import assign_split, chronological_split
from src.graphs.correlation import pearson_knn


def test_pearson_graph_uses_train_rows_only(tiny_refit):
    houses = load_raw_houses(tiny_refit, exclude=[3])
    hourly = pd.concat(
        [watts_to_hourly_kwh(df, min_samples_per_hour=90) for df in houses.values()],
        ignore_index=True,
    )
    clipped, decision = choose_common_window(hourly, coverage_threshold=0.5, min_common_months=0)
    panel = dense_panel(clipped, decision.window_start, decision.window_end)
    panel = impute_short_gaps(panel, max_impute_hours=3)
    split = chronological_split(pd.DatetimeIndex(sorted(panel["timestamp"].unique())))
    panel["split"] = assign_split(panel["timestamp"], split)

    # Poison the test period with a huge shared spike. A leaking graph would
    # pick up a near-perfect correlation from that spike.
    test = panel["split"] == "test"
    panel.loc[test, "kwh"] = 1e6

    adj, corr = pearson_knn(panel, decision.kept, k=2)
    train_only, _ = pearson_knn(
        panel[panel["split"] == "train"].assign(split="train"),
        decision.kept,
        k=2,
    )
    leaking, leaking_corr = pearson_knn(
        panel.assign(split="train"),
        decision.kept,
        k=2,
    )
    assert np.allclose(adj, train_only)
    # A graph that saw the test spike collapses to near-perfect correlation.
    leak_off = leaking_corr.values[np.triu_indices_from(leaking_corr.values, k=1)]
    train_off = corr.values[np.triu_indices_from(corr.values, k=1)]
    assert np.nanmean(np.abs(leak_off)) > np.nanmean(np.abs(train_off))
    assert not np.allclose(adj, leaking)


def test_imputation_does_not_flip_observed_mask():
    idx = pd.date_range("2014-01-01", periods=6, freq="h", tz="UTC")
    panel = pd.DataFrame(
        {
            "household_id": [1] * 6,
            "timestamp": idx,
            "kwh": [1.0, np.nan, np.nan, 4.0, 5.0, 6.0],
            "observed": [True, False, False, True, True, True],
            "n_samples": [100, 0, 0, 100, 100, 100],
        }
    )
    filled = impute_short_gaps(panel, max_impute_hours=3)
    assert filled["observed"].tolist() == panel["observed"].tolist()
    assert np.isfinite(filled["kwh_input"].iloc[1])
