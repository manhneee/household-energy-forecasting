import numpy as np

from src.eval.metrics import per_household_mae_arrays
from src.eval.stats import wilcoxon_household_mae


def test_per_household_mae_arrays():
    # 2 houses, 2 windows each, horizon 2
    y_true = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [20.0, 20.0]])
    y_pred = np.array([[1.0, 1.0], [2.0, 2.0], [11.0, 11.0], [22.0, 22.0]])
    hids = np.array([1, 1, 2, 2])
    mask = np.ones_like(y_true, dtype=bool)
    out = per_household_mae_arrays(y_true, y_pred, hids, mask)
    assert abs(out[1] - 0.0) < 1e-9
    assert abs(out[2] - 1.5) < 1e-9


def test_wilcoxon_household_mae_reports_wins():
    a = np.array([0.2, 0.3, 0.25, 0.4])
    b = np.array([0.3, 0.35, 0.20, 0.5])
    out = wilcoxon_household_mae(a, b)
    assert out["n"] == 4
    assert out["wins_a"] == 3  # a lower on houses 0,1,3
    assert out["wins_b"] == 1
    assert "pvalue" in out
    assert "median_diff_mae" in out
