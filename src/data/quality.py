"""Common window, coverage filter, short-gap imputation, and the quality report.

A graph model needs one dense N x T grid. REFIT houses do not share start/end
dates and mean uptime is ~88%. Decisions here are written to
reports/data_quality.md so Chapter 3 is generated from an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class QualityDecision:
    excluded_solar: list[int]
    dropped_low_coverage: list[int]
    dropped_to_protect_window: list[int]
    kept: list[int]
    window_start: pd.Timestamp | None
    window_end: pd.Timestamp | None
    n_hours: int
    coverage: dict[int, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def per_house_span(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hid, g in hourly.groupby("household_id"):
        valid = g.loc[g["observed"]]
        if valid.empty:
            rows.append(
                {
                    "household_id": hid,
                    "first": pd.NaT,
                    "last": pd.NaT,
                    "n_observed": 0,
                    "n_hours_span": 0,
                    "coverage": 0.0,
                }
            )
            continue
        first = valid["timestamp"].min()
        last = valid["timestamp"].max()
        span_hours = int((last - first) / pd.Timedelta(hours=1)) + 1
        rows.append(
            {
                "household_id": hid,
                "first": first,
                "last": last,
                "n_observed": int(valid.shape[0]),
                "n_hours_span": span_hours,
                "coverage": valid.shape[0] / span_hours if span_hours else 0.0,
            }
        )
    return pd.DataFrame(rows)


def choose_common_window(
    hourly: pd.DataFrame,
    coverage_threshold: float = 0.80,
    min_common_months: int = 12,
) -> tuple[pd.DataFrame, QualityDecision]:
    """Intersect per-house valid spans, drop houses that collapse the window."""
    spans = per_house_span(hourly)
    decision = QualityDecision(
        excluded_solar=[],
        dropped_low_coverage=[],
        dropped_to_protect_window=[],
        kept=[],
        window_start=None,
        window_end=None,
        n_hours=0,
    )

    usable = spans.dropna(subset=["first", "last"]).copy()
    if usable.empty:
        return hourly.iloc[0:0], decision

    kept_ids = set(usable["household_id"].astype(int))

    def _intersection(ids: set[int]) -> tuple[pd.Timestamp, pd.Timestamp]:
        sub = usable[usable["household_id"].isin(ids)]
        return sub["first"].max(), sub["last"].min()

    start, end = _intersection(kept_ids)
    # Drop the house that most shortens the intersection if the window is thin.
    while start < end:
        months = (end - start) / pd.Timedelta(days=30.44)
        if months >= min_common_months:
            break
        # The house with the latest start or earliest end is the bottleneck.
        sub = usable[usable["household_id"].isin(kept_ids)]
        late_start = sub.loc[sub["first"].idxmax()]
        early_end = sub.loc[sub["last"].idxmin()]
        drop_id = int(
            late_start["household_id"]
            if late_start["first"] >= (end - (early_end["last"] - start))
            else early_end["household_id"]
        )
        if drop_id not in kept_ids or len(kept_ids) <= 3:
            decision.notes.append(
                f"Common window is only {months:.1f} months; stopped dropping houses."
            )
            break
        kept_ids.remove(drop_id)
        decision.dropped_to_protect_window.append(drop_id)
        start, end = _intersection(kept_ids)

    if start >= end:
        decision.notes.append("No overlapping window after house drops.")
        return hourly.iloc[0:0], decision

    clipped = hourly[
        hourly["household_id"].isin(kept_ids)
        & (hourly["timestamp"] >= start)
        & (hourly["timestamp"] <= end)
    ].copy()

    coverage = {}
    n_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
    for hid, g in clipped.groupby("household_id"):
        coverage[int(hid)] = float(g["observed"].mean()) if len(g) else 0.0

    low = [hid for hid, cov in coverage.items() if cov < coverage_threshold]
    if low:
        decision.dropped_low_coverage = sorted(low)
        kept_ids -= set(low)
        clipped = clipped[clipped["household_id"].isin(kept_ids)]
        coverage = {hid: coverage[hid] for hid in kept_ids}

    decision.kept = sorted(kept_ids)
    decision.window_start = start
    decision.window_end = end
    decision.n_hours = n_hours
    decision.coverage = coverage
    return clipped, decision


def dense_panel(hourly: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Reindex every kept house onto the same hourly UTC grid."""
    index = pd.date_range(start, end, freq="h", tz="UTC")
    frames = []
    for hid, g in hourly.groupby("household_id"):
        s = g.set_index("timestamp")[["kwh", "observed", "n_samples"]].reindex(index)
        s["kwh"] = pd.to_numeric(s["kwh"], errors="coerce")
        s["observed"] = s["observed"].astype("boolean").fillna(False).astype(bool)
        s["n_samples"] = s["n_samples"].fillna(0).astype(int)
        s["household_id"] = hid
        s = s.reset_index().rename(columns={"index": "timestamp"})
        frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else hourly.iloc[0:0]


def impute_short_gaps(panel: pd.DataFrame, max_impute_hours: int = 3) -> pd.DataFrame:
    """Interpolate short holes for model input. The observed mask is unchanged."""
    out = []
    for _, g in panel.groupby("household_id", sort=True):
        g = g.sort_values("timestamp").copy()
        series = g.set_index("timestamp")["kwh"].astype(float)
        filled = series.interpolate(method="time", limit=max_impute_hours, limit_direction="both")
        g["kwh_input"] = filled.to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else panel


def write_quality_report(
    path: Path,
    decision: QualityDecision,
    spans: pd.DataFrame,
    hourly: pd.DataFrame,
    solar_excluded: list[int],
) -> None:
    decision.excluded_solar = list(solar_excluded)
    path.parent.mkdir(parents=True, exist_ok=True)
    months = 0.0
    if decision.window_start is not None and decision.window_end is not None:
        months = (decision.window_end - decision.window_start) / pd.Timedelta(days=30.44)

    lines = [
        "# REFIT data quality",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Numbers in this file come from the Kaggle files, not from Murray et al. (2017).",
        "Paper values (20 houses, 88% mean uptime) are a prior, not a substitute.",
        "",
        "## Decisions",
        "",
        f"- Solar-contaminated houses excluded from the main set: `{decision.excluded_solar}`",
        f"- Dropped to protect the common window: `{decision.dropped_to_protect_window}`",
        f"- Dropped for coverage < threshold: `{decision.dropped_low_coverage}`",
        f"- Kept households: `{decision.kept}` (N={len(decision.kept)})",
        f"- Common window (UTC): `{decision.window_start}` → `{decision.window_end}`",
        f"- Hours in window: {decision.n_hours} ({months:.1f} months)",
        "",
    ]
    if decision.notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {n}" for n in decision.notes)
        lines.append("")

    lines += [
        "## Per-house coverage inside the common window",
        "",
        "| household_id | coverage |",
        "|---:|---:|",
    ]
    for hid in decision.kept:
        lines.append(f"| {hid} | {decision.coverage.get(hid, float('nan')):.3f} |")

    lines += [
        "",
        "## Per-house raw span (before intersection)",
        "",
        "| household_id | first | last | n_observed | coverage_in_own_span |",
        "|---:|---|---|---:|---:|",
    ]
    for _, row in spans.sort_values("household_id").iterrows():
        lines.append(
            f"| {int(row['household_id'])} | {row['first']} | {row['last']} | "
            f"{int(row['n_observed'])} | {row['coverage']:.3f} |"
        )

    if not hourly.empty:
        zero_share = float((~hourly["observed"]).mean())
        lines += [
            "",
            "## Mask summary (hourly, after aggregation, before dense reindex)",
            "",
            f"- Rows: {len(hourly)}",
            f"- Missing share: {zero_share:.3f}",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
