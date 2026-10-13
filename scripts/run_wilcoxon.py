"""Export per-household MAE and Wilcoxon signed-rank pairs (thesis primary test).

Default models: E01, E04b, E05, E07 at seed 0 (retrains learned models).
Optional: --include-e08 for E08 P1 K=90 (slow; full personalize).

Outputs under experiments/results/:
  household_mae_<id>_seed<slot>.csv
  household_mae_matrix.csv
  wilcoxon_pairs.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_data_config, load_experiment
from src.data.pipeline import load_processed_panel
from src.eval.baselines import evaluate_naive
from src.eval.stats import wilcoxon_household_mae
from src.train.gcgru_run import run_gcgru
from src.train.lstm_run import run_lstm
from src.train.patchtst_run import run_patchtst
from src.train.personalize import run_personalize
from src.train.seeds import set_seed

OUT_DIR = ROOT / "experiments" / "results"

DEFAULT_CONFIGS = {
    "E01": "configs/experiment/e01_seasonal_naive.yaml",
    "E04b": "configs/experiment/e04b_lstm_global_weather.yaml",
    "E05": "configs/experiment/e05_patchtst_global_weather.yaml",
    "E07": "configs/experiment/e07_gcgru_weather.yaml",
    "E08": "configs/experiment/e08_fewshot_lstm.yaml",
}

DEFAULT_PAIRS = [
    ("E01", "E04b"),
    ("E01", "E05"),
    ("E04b", "E05"),
    ("E04b", "E07"),
    ("E05", "E07"),
]


def _series_from_metrics(metrics: dict) -> pd.Series:
    ph = metrics.get("per_house_mae") or {}
    if not ph:
        raise RuntimeError("metrics missing per_house_mae")
    return pd.Series({int(k): float(v) for k, v in ph.items()}, name="mae").sort_index()


def _run_one(exp_id: str, config_path: str, panel, split, data_cfg, seed: int) -> pd.Series:
    exp = load_experiment(config_path)
    set_seed(seed)
    print(f"=== {exp_id} seed={seed} ===", flush=True)
    if exp_id == "E01":
        metrics = evaluate_naive(panel, split, exp, data_cfg)
    elif exp_id == "E04b":
        metrics = run_lstm(panel, split, exp, data_cfg, seed=seed)
    elif exp_id == "E05":
        metrics = run_patchtst(panel, split, exp, data_cfg, seed=seed)
    elif exp_id == "E07":
        metrics = run_gcgru(panel, split, exp, data_cfg, seed=seed)
    elif exp_id == "E08":
        rows = run_personalize(panel, split, exp, data_cfg, seed=seed)
        match = [
            r
            for r in rows
            if r.get("protocol") == "P1" and int(r.get("k_days", -1)) == 90
        ]
        if not match:
            raise RuntimeError("E08: no P1 K=90 row")
        metrics = match[0]
    else:
        raise ValueError(exp_id)
    series = _series_from_metrics(metrics)
    print(
        f"{exp_id} pooled_mae={metrics.get('mae', float('nan')):.4f}  "
        f"mean_house={series.mean():.4f}  n={len(series)}",
        flush=True,
    )
    return series


def _write_house_csv(exp_id: str, seed: int, series: pd.Series) -> Path:
    path = OUT_DIR / f"household_mae_{exp_id}_seed{seed}.csv"
    df = series.rename("mae").reset_index()
    df.columns = ["household_id", "mae"]
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-house MAE + Wilcoxon export")
    parser.add_argument("--seed", type=int, default=0, help="Seed for learned models (default 0).")
    parser.add_argument(
        "--include-e08",
        action="store_true",
        help="Also run E08 P1@K=90 (full personalize; slow).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Subset of experiment ids, e.g. E01 E04b",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wanted = list(args.only) if args.only else ["E01", "E04b", "E05", "E07"]
    if args.include_e08 and "E08" not in wanted:
        wanted.append("E08")

    data_cfg = load_data_config()
    panel, split = load_processed_panel(data_cfg)
    seed = int(args.seed)

    columns: dict[str, pd.Series] = {}
    for exp_id in wanted:
        if exp_id not in DEFAULT_CONFIGS:
            raise SystemExit(f"Unknown experiment {exp_id}; known={list(DEFAULT_CONFIGS)}")
        series = _run_one(exp_id, DEFAULT_CONFIGS[exp_id], panel, split, data_cfg, seed)
        _write_house_csv(exp_id, seed, series)
        label = f"{exp_id}" if exp_id != "E08" else "E08_P1_K90"
        columns[label] = series

    matrix = pd.DataFrame(columns)
    matrix.index.name = "household_id"
    matrix_path = OUT_DIR / "household_mae_matrix.csv"
    matrix.to_csv(matrix_path)
    print(f"wrote {matrix_path}", flush=True)

    pairs = list(DEFAULT_PAIRS)
    if "E08_P1_K90" in matrix.columns:
        pairs.extend(
            [
                ("E04b", "E08_P1_K90"),
                ("E05", "E08_P1_K90"),
                ("E01", "E08_P1_K90"),
            ]
        )

    rows = []
    for a, b in pairs:
        if a not in matrix.columns or b not in matrix.columns:
            continue
        va = matrix[a].to_numpy(dtype=float)
        vb = matrix[b].to_numpy(dtype=float)
        # Align on shared index (already same houses).
        stats = wilcoxon_household_mae(va, vb)
        # Interpret: wins_a = houses where A has lower MAE (A better).
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n": stats["n"],
                "median_diff_a_minus_b": stats["median_diff_mae"],
                "mean_diff_a_minus_b": stats["mean_diff_mae"],
                "wins_a_lower_mae": stats["wins_a"],
                "wins_b_lower_mae": stats["wins_b"],
                "ties": stats["ties"],
                "statistic": stats["statistic"],
                "pvalue": stats["pvalue"],
                "seed": seed,
                "note": "diff = MAE_a - MAE_b; negative median => A better on typical house",
            }
        )
        print(
            f"Wilcoxon {a} vs {b}: p={stats['pvalue']:.4g}  "
            f"median_diff(A-B)={stats['median_diff_mae']:+.4f}  "
            f"wins A/B/tie={stats['wins_a']}/{stats['wins_b']}/{stats['ties']}",
            flush=True,
        )

    pairs_path = OUT_DIR / "wilcoxon_pairs.csv"
    pd.DataFrame(rows).to_csv(pairs_path, index=False)
    print(f"wrote {pairs_path}", flush=True)
    print("Done. N=17 power is low — report wins + median_diff, not only p-value.", flush=True)


if __name__ == "__main__":
    main()
