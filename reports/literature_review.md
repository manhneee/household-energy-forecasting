# Literature review (working notes)

Last updated: 2026-08-22. This is a reading log for Chapter 2, not a polished chapter.

## Position

GNN + weather + household (or building) load forecasting is already published. The remaining Bachelor-sized gap on REFIT is a controlled comparison of Individual vs Global vs household-household Graph, plus leakage-controlled weather and a few-shot personalization curve, with statistical tests across households.

## ST-GNN for residential load

- Nguyen, Delgado Fernandez, and Potenciano Menci (2025), [arXiv:2502.12175](https://arxiv.org/abs/2502.12175) / IEEE PowerTech 2025. Benchmark of GRUGCN, GCGRU, T-GCN, AGCRN, Graph WaveNet, FC-GNN, BP-GNN on Low Carbon London. Graph features help at household level; construction method barely matters; learnable graphs can overfit; **weather was not included**.
- Lin, Wu, and Boulet (2021), *IEEE Trans. Smart Grid*, [10.1109/TSG.2021.3093549](https://doi.org/10.1109/TSG.2021.3093549). Spatial-temporal GNN for residential STLF.
- E3S Web of Conf. (2025), [10.1051/e3sconf/202668702006](https://doi.org/10.1051/e3sconf/202668702006). Building/occupancy GNN + weather, physical / correlation / learned graphs.

## Weather + deep models (no household graph)

- *Energy Informatics* (2026), [10.1186/s42162-026-00635-8](https://doi.org/10.1186/s42162-026-00635-8). Household weather + Transformer–BiLSTM.

## Personalization

- Hypernetwork personalization + weather on ~6,000 Luxembourg households: [arXiv:2506.14472](https://arxiv.org/abs/2506.14472) (2025).
- Personalized / federated household forecasting: *Scientific Reports* (2026), [10.1038/s41598-026-53020-6](https://doi.org/10.1038/s41598-026-53020-6).

## Foundation models and REFIT

- Chronos-2, [arXiv:2510.15821](https://arxiv.org/abs/2510.15821). Native past/future covariates, open fine-tune.
- TiRex-2, [arXiv:2607.01204](https://arxiv.org/abs/2607.01204). Strong and small; public release is zero-shot only.
- Meyer et al. (2025), *IEEE Access*, [10.1109/ACCESS.2025.3648056](https://doi.org/10.1109/ACCESS.2025.3648056). Univariate TSFMs on **hourly REFIT**. They chose REFIT because London is in Chronos/Moirai pretraining. No household graph, no weather covariates, no few-shot curve.
- Exogenous TSFM study: [arXiv:2602.05390](https://arxiv.org/abs/2602.05390).
- US-grid 2026 benchmark [arXiv:2602.21415](https://arxiv.org/abs/2602.21415): PatchTST wins without weather; iTransformer gains more once weather is added.

## REFIT-specific work that is a different question

- Appliance-level GNN on REFIT (nodes = appliances inside one house): IEEE TAI, [10.1109/TAI.2024.3507734](https://doi.org/10.1109/TAI.2024.3507734).
- Single-house REFIT forecasting (SARIMA to TFT): [arXiv:2512.00856](https://arxiv.org/abs/2512.00856).
- Dataset paper: Murray, Stankovic, and Stankovic (2017), *Scientific Data* 4:160122, [10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122).

## Models we are not implementing

EnergyMamba (2026, [arXiv:2606.00506](https://arxiv.org/abs/2606.00506)) is GCN + Mamba on US **grid** data — wrong scale for 17 houses. Informer / Autoformer / FEDformer / TimesNet / TimeMixer / N-BEATS / ASTGCN / DCRNN / GraphSAGE zoos add variance without answering the RQs.
