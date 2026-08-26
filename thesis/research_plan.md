# Research plan

**Working title.** Few-Shot Personalization of Weather-Aware Spatio-Temporal Models for Neighborhood-Scale Household Electricity Forecasting

**Version.** 0.1.0 — 2026-08-22

**Scope.** Forecasting only. Cost optimization is out of scope.

## Questions

1. **RQ1 (supervisor).** Does a weather-aware ST-GNN improve 24-hour household forecasts over a weather-aware LSTM when households share one town and one weather series?
2. **RQ2 (contribution).** How much target-household history (1 / 7 / 30 / 90 days) is required to adapt a global neighborhood model, and does an explicit graph help or hurt that adaptation?
3. **RQ3 (ablation).** After blocking temporal leakage, which weather set (none / temperature / selected) and which graph (none / train-only Pearson / metadata / adaptive) change accuracy?

## Claim language

This thesis provides a leakage-controlled experimental comparison, not a new GNN. Do not claim that GNN + weather + household electricity is novel. Do not claim first forecasting or first TSFM on REFIT.

## Dataset

REFIT via Kaggle `kyleahmurphy/uk-electrical-load` (Murray, Stankovic, and Stankovic, 2017). 20 Loughborough houses, 8-second Watts, 2013–2015. Main experiments exclude solar-contaminated houses 3, 11, 21. Target is hourly `Aggregate` kWh. GNN nodes are households, not appliances.

## Primary method

- Task: lookback 168 h, horizon 24 h, hourly.
- Graph: train-only Pearson kNN (k=3). Metadata similarity as a non-geographic alternative. AGCRN adaptive as an overfitting ablation.
- Models: LSTM 2×2 (individual/global × no-weather/weather), PatchTST, GCGRU/T-GCN.
- Personalization: P0 frozen, P1 head-only, P2 full fine-tune, adaptation window = last K days before test.
- Weather: Open-Meteo ERA5 oracle. Operational 2014 forecasts cannot be reconstructed.

## Success criterion

A complete thesis is a table of Individual vs Global vs Graph, a personalization curve, and a Wilcoxon test across households, with every leakage rule in `DATA_LEAKAGE.md` respected. A negative graph result on N≈17 is valid.

## Chapter map

1. Introduction — neighborhood STLF, RQs
2. Related work — what is already done
3. Data — REFIT, exclusions, common window, UTC rule
4. Method — graphs available on REFIT, models, P0/P1/P2
5. Experiments — matrix, seeds, Kaggle limits
6. Results — tables, curve, significance
7. Discussion — when the graph helps, weather oracle, compute
8. Conclusion

## Plan B / C

- **B.** Drop the GNN; keep LSTM/PatchTST + personalization.
- **C.** E01, E04b, E07, one weather ablation, Wilcoxon. Minimum viable Bachelor thesis.
