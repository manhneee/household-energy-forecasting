from src.data.demo import make_demo_panel
from src.train.personalize import run_personalize


def test_personalize_smoke_returns_rows():
    panel, split = make_demo_panel(n_households=3, n_days=40, seed=3)
    data_cfg = {"task": {"lookback_hours": 12, "horizon_hours": 6}}
    exp = {
        "id": "E08",
        "weather_set": "selected",
        "mode": "personalize",
        "personalization": {
            "protocols": ["P0", "P1"],
            "k_days": [1, 7],
            "max_epochs": 2,
            "lr_scale": 0.1,
        },
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
    rows = run_personalize(panel, split, exp, data_cfg, seed=0)
    assert len(rows) == 4  # 2 protocols × 2 K
    assert all(r["mae"] == r["mae"] for r in rows)  # not NaN
    assert all(r["n_households"] == 3 for r in rows)
