# Research plan

**Working title.** Few-Shot Personalization of Weather-Aware Spatio-Temporal Models for Neighborhood-Scale Household Electricity Forecasting

**Version.** 0.2.4 — 2026-10-13 (Wilcoxon export all 8 pairs including E08 P1@90; RQs unchanged)

**Scope.** Forecasting only. Cost optimization is out of scope.

## Progress (REFIT N=17)

| Block | Status |
|-------|--------|
| E01–E02 baselines | Done |
| LSTM 2×2 (E03–E04b) | Done (3 seeds) |
| E05 PatchTST | Done (mean MAE 0.219) |
| E06–E07 GCGRU + Pearson | Done; **RQ1 negative** (E07 0.230 vs E04b 0.225) |
| E08 P0/P1/P2 on E04b | Done (3 seeds); P1@K=90 mean MAE **0.222** (best adapt) |
| Wilcoxon export (8 pairs, seed 0) | **Done** — `experiments/results/wilcoxon_pairs.csv` |
| E08 per-house Wilcoxon | **Done** (`scripts/run_wilcoxon.py --include-e08`) |
| E09 TSFM, E10 leave-4 | Open |
| Metadata / AGCRN graph ablations | Code / stub only |
| **HPO / hyperparameter tune (light)** | **Deferred** — do when student says so (see below) |

Live numbers: `reports/bao-cao-toan-bo.html`, `experiments/results/experiment_log.csv`.

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

## Hyperparameter tuning (deferred)

**Status.** Planned, **not started**. Run only when the student explicitly asks (“làm HPO / tune”). Do **not** auto-start before Wilcoxon / E09 / E10 unless they say so.

**Intent.** Light retune of the **winning / main** trained models after the comparison matrix is locked — not a full Optuna zoo, not cost optimization (out of scope).

**Suggested scope (confirm at run time):**

- Target models: at least E04b (LSTM global) and E05 (PatchTST); optionally E07 if still in main text.
- Search space (typical): `lr`, `hidden` / `d_model`, `dropout`, `batch_size`, PatchTST `patch_len`/`stride`; keep lookback **168** and horizon **24** fixed.
- Objective: **val MAE** (masked, kWh), train-only scaler/graph rules unchanged (`DATA_LEAKAGE.md`).
- Budget: small (e.g. Optuna ≤20–30 trials or a short grid) so Kaggle / local CPU limits stay feasible.
- Report: best val config → **one** refit on train+val protocol already used, then **test once**; state default vs tuned MAE; do not retune on test.

**Place in schedule.** After Wilcoxon export preferred; can sit in weeks 7–9 next to E09/E10 or as a short appendix experiment. Label in logs as e.g. `E04b-HPO` / `E05-HPO` so default protocol numbers stay comparable.

## Success criterion

A complete thesis is a table of Individual vs Global vs Graph, a personalization curve, and a Wilcoxon test across households, with every leakage rule in `DATA_LEAKAGE.md` respected. A negative graph result on N≈17 is valid. HPO is optional polish, not required for Plan C.

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

- **B.** Drop the GNN; keep LSTM/PatchTST + personalization. **Favoured after RQ1 negative.**
- **C.** E01, E04b, E07, one weather ablation, Wilcoxon. Minimum viable Bachelor thesis (**Wilcoxon seed-0 pairs done**; writing still open).
