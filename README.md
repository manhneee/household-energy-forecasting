# Weather-Aware Neighborhood Household Load Forecasting

Bachelor thesis repository. Working title:

**Few-Shot Personalization of Weather-Aware Spatio-Temporal Models for Neighborhood-Scale Household Electricity Forecasting**

This is a leakage-controlled comparison of Individual vs Global vs Graph vs few-shot personalization on the UK REFIT dataset (20 Loughborough households, 2013–2015), with Open-Meteo weather. The thesis does **not** claim that GNN + weather + household forecasting is novel.

Dataset: [kyleahmurphy/uk-electrical-load](https://www.kaggle.com/datasets/kyleahmurphy/uk-electrical-load) (REFIT; Murray, Stankovic, and Stankovic, 2017, *Scientific Data* 4:160122).

Current version: see [`VERSION`](VERSION).

Giải thích source (làm gì / vì sao): [`docs/hieu-source/00-doc-tu-day.md`](docs/hieu-source/00-doc-tu-day.md).

---

## Research questions

- **RQ1.** Does a weather-aware ST-GNN improve 24-hour household forecasts over a weather-aware LSTM when households share one town and one weather series?
- **RQ2.** How much target-household history (1 / 7 / 30 / 90 days) is required to adapt a global neighborhood model, and does an explicit graph help or hurt that adaptation?
- **RQ3.** After blocking temporal leakage, which weather set and which graph change accuracy?

---

## What this repo will and will not do

**Will:** hourly household `Aggregate` forecasting, lookback 168 h, horizon 24 h, train-only scalers and graphs, observation-masked metrics, 3 random seeds, Wilcoxon tests across households.

**Will not:** cost optimization, appliance-level GNN nodes, raw 8-second training, a zoo of Transformers / TSFMs, or inflating N with the London dataset.

---

## Dataset facts that the code must respect

- 20 houses, files named `CLEAN_House1.csv` … `CLEAN_House21.csv`. **There is no House 14.** The loader reads the file list; it never uses `range(1, 21)`.
- Main experiments **exclude solar-contaminated Houses 3, 11, 21**. Houses 1, 6, 7 were rewired and stay in.
- REFIT `DateTime` is British Summer Time–corrected wall-clock time. Open-Meteo `/v1/archive` defaults to UTC. Both sides are converted to UTC before the merge. See [`DATA_LEAKAGE.md`](DATA_LEAKAGE.md).
- Cleaned REFIT fills gaps longer than 2 minutes with zeros. Long exact-zero hours are treated as missing, not as genuine zero load.

---

## Project layout

```text
configs/            data.yaml, weather.yaml, model/*.yaml, experiment/*.yaml
src/                importable library — notebooks stay thin
  data/             load, aggregate, quality, split, window, features
  weather/          Open-Meteo cache + UTC check
  graphs/           Pearson, metadata similarity, adaptive stub
  models/           naive now; LSTM / PatchTST / GCGRU later
  train/            seeds, experiment log
  eval/             masked metrics, plots
notebooks/          Kaggle-runnable, import from src/
experiments/results/experiment_log.csv
reports/            literature, data quality, figures
thesis/             research_plan.md
```

---

## Local setup

```bash
cd household-energy-forecasting
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Put the Kaggle REFIT files under `data/raw/` (or set `paths.raw_dir` in `configs/data.yaml`):

```text
data/raw/CLEAN_House1.csv
data/raw/CLEAN_House2.csv
...
```

Download options:

```bash
# if the Kaggle CLI is installed and authenticated
kaggle datasets download -d kyleahmurphy/uk-electrical-load -p data/raw --unzip
```

If the REFIT zip is not downloaded yet, build a synthetic neighborhood panel and smoke-test the LSTM:

```bash
python scripts/make_demo_panel.py
python scripts/run_experiment.py --config configs/experiment/e01_seasonal_naive.yaml
python scripts/run_experiment.py --config configs/experiment/e04b_lstm_smoke.yaml
```

With the real files in `data/raw/`:

```bash
python scripts/build_panel.py
python scripts/run_eda.py
pytest -q
```

`build_panel.py` writes `data/processed/hourly_panel.parquet`, the observation mask, the quality report, and the weather cache.

---

## Kaggle setup

1. Create a notebook and **Add data** → `kyleahmurphy/uk-electrical-load`.
2. Upload this repository as a dataset, or clone it into `/kaggle/working`.
3. Confirm the input folder. The default in `configs/data.yaml` is `/kaggle/input/uk-electrical-load`. If Kaggle mounted a nested folder, set `paths.raw_dir` to the directory that actually contains `CLEAN_House*.csv`.
4. Run `notebooks/01_data_exploration.ipynb` top to bottom.
5. Expected outputs:
   - `reports/data_quality.md` with the common window, coverage table, and dropped houses
   - figures under `reports/figures/`
   - weather cache under `data/weather/` (or `/kaggle/working/data/weather`)

Internet must be **on** the first time weather is fetched. After that the cache is reused.

Kaggle limits: 12 hours CPU / 12 hours GPU, ~30 GB disk, P100/T4 GPU. Hourly N≈17 is well inside those limits. Do not fine-tune Chronos-2 120M on this machine.

---

## Experiment matrix (Plan A)

| ID | Model | Weather | Graph |
|----|-------|---------|-------|
| E01 | Seasonal naive (previous day) | no | no |
| E02 | Persistence (1-hour diagnostic) | no | no |
| E03 | LSTM individual | no | no |
| E03b | LSTM individual | yes | no |
| E04 | LSTM global | no | no |
| E04b | LSTM global | yes | no |
| E05 | PatchTST global | yes | no |
| E06 | GCGRU / T-GCN | no | train-only Pearson |
| E07 | GCGRU / T-GCN | yes | train-only Pearson |
| E08 | Few-shot P0/P1/P2 of E04b and E07 | yes | as source |
| E09 | One TSFM (Chronos-2, else TiRex-2, else TTM) | yes if supported | no |
| E10 | Leave-4-households-out transfer | yes | Pearson |

Every trained model uses seeds `{0, 1, 2}`. Naive baselines run once.

Run an experiment from its YAML:

```bash
python scripts/run_experiment.py --config configs/experiment/e01_seasonal_naive.yaml
```

---

## Evaluation rules (short)

- Split is chronological 70 / 15 / 15 by time, never by row shuffle.
- Scalers and the Pearson graph are fit on **train timestamps only**.
- Metrics (MAE, RMSE, sMAPE, WAPE, MASE) are computed in **kWh on observed hours only**.
- Do not lead with MAPE. Zero and near-zero hours break it.
- Primary statistical test: Wilcoxon signed-rank on per-household MAE.

Full protocol: [`DATA_LEAKAGE.md`](DATA_LEAKAGE.md).

---

## Citation

Murray, D., Stankovic, L. & Stankovic, V. An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. *Sci. Data* **4**, 160122 (2017). https://doi.org/10.1038/sdata.2016.122

Weather: Open-Meteo Historical Weather API (ERA5), https://open-meteo.com/en/docs/historical-weather-api
