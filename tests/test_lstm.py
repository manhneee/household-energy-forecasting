import pandas as pd

from src.data.demo import make_demo_panel
from src.train.lstm_run import run_lstm


def test_global_lstm_smoke_returns_finite_mae():
    panel, split = make_demo_panel(n_households=3, n_days=20, seed=1)
    data_cfg = {
        "task": {"lookback_hours": 12, "horizon_hours": 6},
    }
    exp = {
        "id": "E04b",
        "weather_set": "selected",
        "mode": "global",
        "model": {
            "kind": "lstm",
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "household_embedding_dim": 2,
            "lr": 0.01,
            "batch_size": 32,
            "max_epochs": 2,
            "patience": 2,
        },
    }
    metrics = run_lstm(panel, split, exp, data_cfg, seed=0)
    assert pd.notna(metrics["mae"])
    assert metrics["mae"] > 0
    assert metrics["n_households"] == 3
