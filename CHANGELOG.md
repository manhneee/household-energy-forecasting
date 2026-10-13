# Changelog

All notable changes to this project are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/).

## [0.3.0] - 2026-09-17

### Added

- PatchTST global (E05) and GCGRU / T-GCN on train-only Pearson kNN (E06, E07).
- Few-shot personalization P0/P1/P2 (E08) on pretrained E04b; K ∈ {1,7,30,90}.
- Graph-synchronized windows (`src/data/graph_dataset.py`).

### Results (REFIT N=17, aggregate MAE, observed hours)

- E04b LSTM global + weather: 0.225 (3 seeds). E05 PatchTST: 0.219.
- E07 GCGRU + weather: 0.230 — does not beat E04b (RQ1 negative on N=17).
- E08 seed 0: P1 at K=90 best (0.221); P2 at K=1 slightly worse than frozen (0.227).

## [0.2.0] - 2026-09-01

### Added

- Weather-aware LSTM for the Individual / Global × no-weather / weather grid (E03, E03b, E04, E04b).
- Train-only per-house load scaler and shared weather scaler.
- Window dataset, masked MSE, early stopping on val MAE in kWh.
- Synthetic demo panel (`scripts/make_demo_panel.py`) so training can start before the REFIT download.

## [0.1.0] - 2026-08-22

### Added

- Project scaffold for the Bachelor thesis on neighborhood-scale household load forecasting.
- REFIT loader that discovers `CLEAN_House*.csv` files instead of assuming IDs `1..20`.
- Hourly aggregation of 8-second Watts to kWh with a sample-count guard.
- Common-window / coverage / observation-mask quality protocol.
- Cached Open-Meteo Historical Weather client for Loughborough, with UTC alignment check.
- Train-only chronological split, seed utility, masked metrics, experiment log.
- Seasonal-naive and persistence baselines (E01, E02).
- Household metadata transcribed from Murray et al. (2017) Table 2 for the metadata-similarity graph.
- Literature review, research plan, and leakage protocol documents.
