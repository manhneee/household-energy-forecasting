# Data leakage protocol

This document is the source of truth for what is allowed to enter training, adaptation, and evaluation. If code and this file disagree, fix the code.

## 1. Temporal split

```text
|-------- train 70% --------|--- val 15% ---|---- test 15% ----|
                            t_train_end     t_val_end
```

- Split is on the **common hourly timeline**, not per household and not by shuffled rows.
- A window whose forecast horizon crosses a split boundary is dropped from that split.
- Validation is used for early stopping and model selection only.
- Test is used once, after the model (or the personalization protocol) is frozen.



## 2. Fit-on-train-only

These objects are estimated on train timestamps of the kept houses, then frozen:

- load scaler (and weather scaler, if used)
- Pearson correlation graph
- any other statistic that looks at more than one household (e.g. global mean for MASE)

They must not be refit on val/test, and they must not be refit after seeing the adaptation window of a target household if that window sits after `t_train_end`. For personalization, see section 5.

## 3. Weather

- Past weather (lookback): Open-Meteo Historical Weather API `/v1/archive` (ERA5 reanalysis).
- Future weather (horizon): the **same ERA5 series**. This is an **oracle**. REFIT is 2013–2015; Open-Meteo Historical Forecast API starts around 2021–2022, so a 2014 operational forecast cannot be reconstructed. Chapter 7 must state this.
- Do not use weather variables that were not listed in `configs/weather.yaml` for that run.
- Timezone: localize REFIT `DateTime` as `Europe/London` (the cleaned release is BST-corrected wall-clock time), convert to UTC, request Open-Meteo in UTC, merge on the UTC hour. After merging, the daily `shortwave_radiation` peak must fall near local noon. A one-hour summer shift is a silent bug.



## 4. Observation mask vs imputation

- Hours that fail the sample-count guard, or that are exact-zero after the cleaned-release zero-fill, are **missing**.
- Short gaps (default ≤ 3 hours) may be linearly interpolated for **model input** so that 168-hour windows stay contiguous.
- The boolean mask `observed` stays False on those hours.
- **Every metric is computed only where** `observed` **is True on the target hours.** Imputed targets never enter MAE / RMSE / sMAPE / WAPE / MASE.



## 5. Personalization (RQ2)

```text
|-------- train --------|--- val ---|-- adapt K --|---- test ----|
                                     last K days before test
```

- The adaptation window is the K days immediately before the test period. It must not overlap test.
- K ∈ {1, 7, 30, 90}.
- **P0** frozen global: no update.
- **P1** head-only: encoder frozen; output layer and household embedding updated.
- **P2** full fine-tune: all weights, learning rate = 1/10 of pretraining.
- Early stopping for P1/P2 uses the last 20% of the adaptation window, never test.
- With lookback 168 h, “K = 1 day” means 24 **target** hours. The input context reaches back before the adaptation window, into val/late-train. That context is allowed (it is past relative to the target). Using any test hour as input or as an early-stopping signal is not.

If K = 1 cannot form a valid window after masking, report the smallest feasible K instead of inventing a shorter lookback for that cell only.

## 6. Household hold-out (E10)

- Four houses are removed from pretraining.
- The Pearson graph for E10 is rebuilt on the remaining training houses and train timestamps.
- E10 numbers are not comparable to the main table (training N drops from ~17 to ~13).



## 7. Forbidden

- Shuffling rows or windows across time.
- Fitting the graph or scaler on the full series “because N is small”.
- Training on Houses 3, 11, 21 in the main table (PV on the aggregate).
- Using appliance channels as GNN nodes.
- Reporting metrics on scaled units or on imputed targets.
- Using MAPE as the lead metric.
- Peeking at test to choose weather variables, k for kNN, or the personalization method.

