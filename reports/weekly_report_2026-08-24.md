# Weekly progress report

**To:** Advisor  
**From:** Household energy forecasting thesis  
**Period:** 17–24 August 2026  
**Repo version:** 0.2.0 (22 August 2026)  
**Working title:** Few-Shot Personalization of Weather-Aware Spatio-Temporal Models for Neighborhood-Scale Household Electricity Forecasting

Vietnamese version: [`weekly_report_2026-08-24.vi.md`](weekly_report_2026-08-24.vi.md)

---

## 1. Status in one paragraph

This week the project moved from a research plan to a **runnable, leakage-controlled pipeline on the real REFIT files**. The hourly panel, weather merge, quality protocol, and seasonal-naive baseline are measured on 17 Loughborough households. The weather-aware LSTM is implemented and smoke-tested on a synthetic 6-house panel; it has **not** yet been trained on REFIT. I am not claiming a model ranking. The numbers below are data facts and a single naive baseline.

---

## 2. What was completed

| Area | Done this week |
|------|----------------|
| Scope | Locked three RQs, experiment matrix E01–E10, and a written leakage protocol (`DATA_LEAKAGE.md`). Claim language: comparison, not a new GNN. |
| Data | Loader discovers `CLEAN_House*.csv` (no House 14). 8-second Watts → hourly kWh with a sample-count guard. Exact-zero hours treated as missing (cleaned REFIT zero-fills gaps). |
| Quality | Common-window protocol on the Kaggle files, not on the 2017 paper averages. Quality report written from the actual files. |
| Weather | Open-Meteo ERA5 for Loughborough (52.77°N, 1.21°W): temperature, shortwave radiation, relative humidity. REFIT localized as Europe/London → UTC before merge. Solar-noon check passed. |
| Split | Chronological 70 / 15 / 15 on the common timeline. Scalers and graphs specified as train-only. |
| Models | Seasonal naive (E01) run on REFIT. Persistence (E02) coded, not yet logged. LSTM Individual/Global × weather/no-weather coded; smoke run on a demo panel only. |
| Graphs | Train-only Pearson kNN (k=3) builder exists. Metadata table transcribed from Murray et al. (2017) Table 2. Adaptive graph and GCGRU/PatchTST are stubs. |
| Tests | Leakage, mask metrics, House 14, zero-hour missing, sample-count, train-only scaler, LSTM smoke (2 epochs on demo). |
| Reading | Working literature log: ST-GNN residential load, weather + deep models, TSFMs on hourly REFIT, personalization. Position: GNN + weather + household load is already published. |

---

## 3. Data analysis (what I would show in the meeting)

### 3.1 Cohort after exclusions

- **Discovered:** 20 houses (IDs 1–13, 15–21). House 14 does not exist.
- **Excluded from the main table:** Houses 3, 11, 21 (PV on the aggregate). Houses 1, 6, 7 were rewired and **stay in**.
- **Kept:** 17 houses. No house was dropped for coverage or for collapsing the common window.
- **Common window (UTC):** 2014-03-20 12:00 → 2015-05-10 23:00 = **9,996 hours (13.7 months)**.
- **Coverage inside that window:** 0.993–0.999 (mean ≈ 0.997). Lowest: House 13.
- **Missing share** after aggregation, before dense reindex: **1.3%**.
- Own-span coverage is worse (House 2: 0.758; House 13: 0.783). The intersection is what training uses; the paper’s “88% mean uptime” is a prior, not a substitute for these file-level counts.

**Implication:** N = 17 is small for a GNN. A negative graph result is a valid Bachelor outcome (already written into the research plan). The common window is long enough for 168 h lookback / 24 h horizon and a chronological test period of about two months.

### 3.2 Chronological split (no row shuffle)

| Split | End timestamp (UTC) | Role |
|-------|---------------------|------|
| Train (70%) | 2015-01-06 00:00 | Fit scalers, graph, model |
| Val (15%) | 2015-03-09 11:00 | Early stopping / selection only |
| Test (15%) | 2015-05-10 23:00 | Frozen evaluation |

Windows whose 24-hour horizon crosses a split boundary are dropped. Test weather on the horizon is **ERA5 oracle** (Open-Meteo Historical Forecast does not cover 2014). Chapter 7 must state this; it is not an operational day-ahead setup.

Weather alignment check after merge: mean local shortwave peak hour = **12.99** (expected 12, tolerance ±2). **Pass.** A one-hour summer shift would have been a silent UTC bug.

### 3.3 Load heterogeneity (observed hours only)

Mean hourly consumption on the processed panel is about **0.49 kWh/h**, but houses are not interchangeable:

| Household | Occupancy | Mean kWh/h | Median kWh/h | Max kWh/h |
|----------:|----------:|-----------:|-------------:|----------:|
| 15 | 1 | 0.257 | 0.187 | 4.04 |
| 19 | 4 | 0.293 | 0.220 | 2.00 |
| 4 | 2 | 0.366 | 0.322 | 3.49 |
| 12 | 3 | 0.374 | 0.227 | 4.54 |
| 20 | 2 | 0.377 | 0.287 | 2.43 |
| 17 | 3 | 0.413 | 0.247 | 4.63 |
| 1 | 2 | 0.439 | 0.280 | 12.51 |
| 18 | 2 | 0.449 | 0.361 | 4.96 |
| 2 | 4 | 0.470 | 0.211 | 5.10 |
| 6 | 2 | 0.470 | 0.406 | 2.96 |
| 7 | 4 | 0.533 | 0.231 | 4.41 |
| 13 | 4 | 0.565 | 0.341 | 4.46 |
| 16 | 6 | 0.569 | 0.436 | 6.01 |
| 9 | 2 | 0.584 | 0.255 | 5.82 |
| 8 | 2 | 0.665 | 0.426 | 6.11 |
| 10 | 4 | 0.709 | 0.435 | 5.08 |
| 5 | 4 | 0.759 | 0.581 | 8.37 |

**Readings I will defend:**

1. **Occupancy is a weak proxy.** House 15 (1 occupant) is the lowest mean, as expected, but House 19 (4 occupants) is almost as low, and House 8 (2 occupants) is among the highest. A global model without a household embedding (or graph) will smear these levels together.
2. **Right-skew / spikes.** Median << mean for several houses (House 7: 0.23 vs 0.53; House 1 max 12.5 kWh). RMSE will look worse than MAE. That is why the protocol leads with MAE / WAPE / MASE, not MAPE.
3. **This is why RQ2 exists.** A neighborhood GNN on N = 17 can overfit a few high-load houses. Few-shot personalization (1 / 7 / 30 / 90 days) is the contribution that does not require claiming a new architecture.

### 3.4 Baseline on REFIT (E01 only)

Seasonal naive = copy the previous day’s 24 hours. Metrics on **observed test hours only**, in kWh.

| Experiment | Data | N houses | MAE | RMSE | sMAPE | WAPE | MASE |
|------------|------|---------:|----:|-----:|------:|-----:|-----:|
| E01 seasonal naive | REFIT, common window | 17 | **0.255** | 0.500 | 39.8 | 0.537 | 1.007 |
| E01 (sanity) | Synthetic 6-house demo | 6 | 0.051 | 0.064 | 11.6 | 0.080 | 0.696 |
| E04b LSTM smoke | Synthetic 6-house demo | 6 | 0.037 | 0.047 | 8.5 | 0.058 | 0.507 |

**Interpretation of E01 on REFIT:**

- MAE **0.255 kWh** against a cohort mean of ~0.49 kWh is a **WAPE of 54%**. Yesterday’s profile is a serious baseline for daily-cyclic residential load; a learned model must beat this on the same masked test windows.
- MASE ≈ **1.00** is expected: the in-sample seasonal scale and the test seasonal-naive error are the same family of forecast.
- RMSE ≈ 2× MAE confirms the spike tail seen in the per-house maxima.
- The LSTM number **0.037 is not comparable** to 0.255. It is a 2-epoch-class smoke run on fake data (16,336 parameters, ~17 s). I will not put it in a results table.

E02 (persistence, 1-hour diagnostic) is implemented but not yet logged on REFIT.

---

## 4. Method choices that need advisor sign-off (already in code)

These are not results. They are decisions that later tables will be uninterpretable without.

1. **One weather series for the whole town.** All 17 houses share Loughborough ERA5. RQ1 is exactly: does an ST-GNN still help when there is no spatial weather variation?
2. **Horizon weather is oracle ERA5.** Honest for 2013–2015; not a 2014 operational forecast.
3. **Metrics never see imputed targets.** Short gaps (≤ 3 h) may be interpolated for model *input*; `observed = False` hours are excluded from MAE/RMSE/sMAPE/WAPE/MASE.
4. **LSTM 2×2 grid** (individual/global × no-weather/weather) before any GNN, so weather and pooling are not confounded.
5. **Plan B / C** already written: drop the GNN if it does not train; minimum viable thesis is E01 + E04b + E07 + one weather ablation + Wilcoxon across households.

---

## 5. Experiment matrix — honest status

| ID | Model | Weather | Graph | Code | Number on REFIT |
|----|-------|---------|-------|------|-----------------|
| E01 | Seasonal naive | no | no | yes | MAE 0.255 kWh |
| E02 | Persistence | no | no | yes | not logged |
| E03 / E03b | LSTM individual | no / yes | no | yes | no |
| E04 / E04b | LSTM global | no / yes | no | yes | demo only |
| E05 | PatchTST | yes | no | stub | — |
| E06 / E07 | GCGRU / T-GCN | no / yes | Pearson | stub | — |
| E08 | Few-shot P0/P1/P2 | yes | as source | protocol only | — |
| E09 | One TSFM | if supported | no | no | — |
| E10 | Leave-4-out | yes | Pearson | YAML list only | — |

E10 holdout placeholder (to confirm): Houses **2, 8, 16, 20** (occupancy 4 / 2 / 6 / 2; mixed load). Chosen so the four houses are not all the same size.

---

## 6. Proposed work for next week

1. Run **E02** on the same REFIT panel and log it next to E01.
2. Train **E04b** (global weather-aware LSTM) on REFIT, seeds {0, 1, 2}, report per-household MAE vs E01 (Wilcoxon when N = 17 pairs exist).
3. If compute allows: **E03b** so individual vs global is not confounded with weather.
4. Produce the four EDA figures (missing heatmap, daily profile, train Pearson, load vs temperature) — the plotting code exists; the PNGs are not in the repo yet.
5. Do **not** start GCGRU until the LSTM 2×2 cells have REFIT numbers. Otherwise we cannot attribute any gain.

---

## 7. Questions for the advisor

1. Is **ERA5-oracle** horizon weather acceptable if it is labelled as such in the thesis, or should weather-on-horizon be ablated to “none” as the operational proxy?
2. With N = 17, should Plan A (GNN + few-shot) stay the default, or should I treat **Plan B** (LSTM/PatchTST + personalization, no GNN) as the main path and keep E07 as an appendix?
3. Please confirm the **E10 holdout** set {2, 8, 16, 20} now that coverage is known, or name four other houses.
4. Is a **negative graph result** (GNN does not beat weather-aware LSTM) acceptable as the RQ1 answer, given the published London ST-GNN benchmark did not include weather?

---

## 8. What I am not claiming

- That the LSTM is better than seasonal naive on REFIT (not trained yet).
- That GNN + weather + household forecasting is novel (literature already has this combination; London ST-GNN work did not use weather, which is the remaining comparison).
- That 0.037 MAE from the demo smoke run is a thesis result.
