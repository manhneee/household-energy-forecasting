"""Produce the required EDA figures from the processed panel."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_data_config
from src.data.pipeline import load_processed_panel
from src.eval.plots import (
    plot_corr_heatmap,
    plot_daily_profile,
    plot_load_vs_temperature,
    plot_missing_heatmap,
)
from src.graphs.correlation import pearson_knn
from src.paths import figures_dir


def main() -> None:
    cfg = load_data_config()
    panel, _ = load_processed_panel(cfg)
    fig = figures_dir(cfg)
    kept = sorted(panel["household_id"].unique().tolist())
    plot_missing_heatmap(panel, fig / "missing_heatmap.png")
    plot_daily_profile(panel, fig / "daily_profile.png")
    adj, corr = pearson_knn(panel, kept, k=3)
    plot_corr_heatmap(corr, fig / "train_pearson.png")
    plot_load_vs_temperature(panel, fig / "load_vs_temperature.png")
    summary = (
        panel.loc[panel["observed"]]
        .groupby("household_id")["kwh"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    summary.to_csv(fig / "household_summary.csv", index=False)
    print(f"Wrote figures to {fig}")
    print(f"Pearson graph shape {adj.shape}, mean edge weight {adj.mean():.4f}")


if __name__ == "__main__":
    main()
