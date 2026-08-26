"""Run one experiment YAML. v0.2.0: E01, E02, and the LSTM 2x2 (E03–E04b)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_data_config, load_experiment
from src.data.pipeline import load_processed_panel
from src.eval.baselines import evaluate_naive
from src.train.experiment_log import append_result, config_hash
from src.train.lstm_run import run_lstm
from src.train.seeds import set_seed


def _kind(exp: dict) -> str:
    model = exp.get("model")
    if isinstance(model, dict):
        return str(model.get("kind") or model.get("name") or "")
    return str(model or "")


def run_one(exp: dict, data_cfg: dict, seed: int, config_path: str) -> dict:
    set_seed(int(seed))
    panel, split = load_processed_panel(data_cfg)
    kind = _kind(exp)
    t0 = time.perf_counter()
    if kind in {"naive", "persistence"} or exp.get("id") in {"E01", "E02"}:
        metrics = evaluate_naive(panel, split, exp, data_cfg)
        metrics["train_seconds"] = 0.0
        metrics["n_params"] = 0
        metrics["n_households"] = int(panel["household_id"].nunique())
    elif kind == "lstm":
        metrics = run_lstm(panel, split, exp, data_cfg, seed=int(seed))
    else:
        raise NotImplementedError(
            f"{exp.get('id')} ({kind}) is not implemented yet. "
            "v0.2.0 covers E01, E02, and E03–E04b."
        )
    elapsed = time.perf_counter() - t0
    row = {
        "experiment_id": exp["id"],
        "seed": seed,
        "config_path": config_path,
        "config_hash": config_hash(exp),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "smape": metrics.get("smape"),
        "wape": metrics.get("wape"),
        "mase": metrics.get("mase"),
        "n_households": metrics.get("n_households"),
        "train_seconds": round(float(metrics.get("train_seconds", 0.0)), 3),
        "infer_seconds": round(elapsed - float(metrics.get("train_seconds", 0.0)), 3),
        "n_params": metrics.get("n_params", 0),
        "notes": exp.get("description", ""),
    }
    append_result(row)
    print(exp["id"], f"seed={seed}", {k: metrics.get(k) for k in ("mae", "rmse", "wape", "mase")})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="If omitted, run every seed in the YAML.")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    exp = load_experiment(args.config)
    if args.max_epochs is not None and isinstance(exp.get("model"), dict):
        exp["model"]["max_epochs"] = args.max_epochs
    data_cfg = load_data_config()
    seeds = [args.seed] if args.seed is not None else list(exp.get("seeds") or [0])
    for seed in seeds:
        run_one(exp, data_cfg, seed, args.config)


if __name__ == "__main__":
    main()
