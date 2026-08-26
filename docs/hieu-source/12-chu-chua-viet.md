# Chỗ chưa viết (đừng nhầm với "đã xong")

Code hiện tại đủ để **hiểu protocol và LSTM**. Đây không phải thesis hoàn chỉnh.

## Stub (`raise NotImplementedError`)

- `src/models/patchtst.py`
- `src/models/gcgru.py`
- `src/graphs/adaptive.py`

YAML E05–E07 trỏ vào chúng. Chạy `run_experiment.py` với các file đó sẽ lỗi cố ý.

## Có hàm / protocol, chưa có vòng lặp

- Personalization P0/P1/P2: luật trong `DATA_LEAKAGE.md`, `adaptation_window()` đã có, `head_parameters()` đã tách. Chưa có `src/train/personalize.py`.
- E10: list nhà trong YAML, chưa pipeline hold-out.
- Diebold–Mariano: chưa.
- Optuna: chưa (chỉ hai model cuối, budget nhỏ).
- TSFM Chronos-2 / TiRex-2 / TTM: chưa.

## Đã chạy vs mới có code

| Việc | Code | Số trên REFIT |
|------|------|----------------|
| Panel hourly + weather | có | có (`data_quality.md`) |
| E01 | có | MAE 0.255 kWh |
| E02 | có | chưa |
| LSTM 2×2 | có | chưa (chỉ demo 6 nhà) |
| Pearson builder | có | heatmap EDA |
| GCGRU train | không | không |
| Few-shot curve | không | không |

## Test hiện có kiểm tra gì

Leak graph, mask metric, House 14, zero-hour missing, sample-count, scaler bỏ test, LSTM smoke 2 epoch trên demo. Test **không** chứng minh LSTM thắng trên REFIT.

## Nên đọc tiếp trong repo (ngoài thư mục này)

- `DATA_LEAKAGE.md` — luật
- `thesis/research_plan.md` — RQ và chapter map
- `reports/literature_review.md` — paper đã có
- `reports/data_quality.md` — số đo trên file Kaggle
- `src/data/pipeline.py` — đọc song song với `03-du-lieu-refit.md`
