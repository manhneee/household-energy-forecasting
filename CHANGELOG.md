# Changelog

All notable changes to this project are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/).

## [0.2.0] - 2026-08-22

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
