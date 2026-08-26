import numpy as np

from src.eval.metrics import mae, masked_metrics


def test_mask_drops_imputed_targets():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 100.0, 3.0])
    mask = np.array([True, False, True])
    assert mae(y_true, y_pred, mask) == 0.0


def test_metrics_are_in_native_units():
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    out = masked_metrics(y_true, y_pred)
    assert abs(out["mae"] - 1.0) < 1e-9
    assert abs(out["rmse"] - 1.0) < 1e-9
    assert abs(out["wape"] - 0.5) < 1e-9
