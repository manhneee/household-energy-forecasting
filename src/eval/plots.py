"""EDA and thesis figures. Saved under reports/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_missing_heatmap(panel: pd.DataFrame, path: Path) -> None:
    wide = panel.pivot(index="timestamp", columns="household_id", values="observed")
    # Daily resolution so a 20-month heatmap stays readable.
    daily = (~wide.astype(bool)).resample("D").mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(daily.T, cmap="mako", cbar_kws={"label": "missing share"})
    plt.title("Missing hours by household (daily share)")
    plt.xlabel("date")
    savefig(path)


def plot_daily_profile(panel: pd.DataFrame, path: Path, local_tz: str = "Europe/London") -> None:
    work = panel.loc[panel["observed"], ["timestamp", "household_id", "kwh"]].copy()
    work["hour"] = work["timestamp"].dt.tz_convert(local_tz).dt.hour
    mean = work.groupby(["household_id", "hour"])["kwh"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=mean, x="hour", y="kwh", hue="household_id", legend=False, alpha=0.35)
    overall = work.groupby("hour")["kwh"].mean()
    plt.plot(overall.index, overall.values, color="black", linewidth=2.5, label="mean")
    plt.legend()
    plt.title("Hour-of-day load profile (observed hours, kWh)")
    savefig(path)


def plot_corr_heatmap(corr: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1)
    plt.title("Train-only Pearson correlation of hourly kWh")
    savefig(path)


def plot_load_vs_temperature(panel: pd.DataFrame, path: Path) -> None:
    if "temperature_2m" not in panel.columns:
        return
    work = panel.loc[panel["observed"], ["household_id", "kwh", "temperature_2m"]]
    g = sns.lmplot(
        data=work,
        x="temperature_2m",
        y="kwh",
        col="household_id",
        col_wrap=4,
        scatter_kws={"s": 6, "alpha": 0.15},
        line_kws={"color": "crimson"},
        height=2.4,
    )
    g.fig.suptitle("Observed load vs ERA5 temperature (evidence for RQ3)", y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(path, dpi=150)
    plt.close()
