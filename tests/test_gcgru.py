import pandas as pd

from src.data.demo import make_demo_panel
from src.train.gcgru_run import run_gcgru


def test_gcgru_smoke_returns_finite_mae():
    panel, split = make_demo_panel(n_households=4, n_days=20, seed=2)
    data_cfg = {"task": {"lookback_hours": 12, "horizon_hours": 6}}
    exp = {
        "id": "E07",
        "weather_set": "selected",
        "mode": "graph",
        "graph": "pearson_knn",
        "model": {
            "kind": "gcgru",
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "lr": 0.01,
            "batch_size": 8,
            "max_epochs": 2,
            "patience": 2,
            "graph": {"type": "pearson_knn", "k": 2},
        },
    }
    metrics = run_gcgru(panel, split, exp, data_cfg, seed=0)
    assert pd.notna(metrics["mae"])
    assert metrics["mae"] > 0
    assert metrics["n_households"] == 4
