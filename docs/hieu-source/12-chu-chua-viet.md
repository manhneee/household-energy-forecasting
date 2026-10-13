# Chỗ chưa viết (đừng nhầm với "đã xong")

Code + số REFIT đã có tới **E08 đủ 3 seed** + **Wilcoxon seed 0**. Còn E09/E10, E08 Wilcoxon (tuỳ chọn), chương viết.

## Stub (`raise NotImplementedError`)

- `src/graphs/adaptive.py` (AGCRN ablation)

## Có hàm / protocol, chưa có vòng lặp

- E10: list nhà trong YAML, chưa pipeline hold-out.
- Diebold–Mariano: chưa.
- Optuna / HPO nhẹ (tune lr, hidden, dropout… trên E04b/E05): **đã ghi plan, hoãn** — chỉ làm khi SV bảo. Xem `thesis/research_plan.md` § Hyperparameter tuning.
- TSFM Chronos-2 / TiRex-2 / TTM: chưa.
- Wilcoxon E01/E04b/E05/E07 seed 0: **có** (`scripts/run_wilcoxon.py` → `wilcoxon_pairs.csv`). E08 per-house: chưa (`--include-e08`).

## Đã chạy vs mới có code

| Việc | Code | Số trên REFIT |
|------|------|----------------|
| Panel hourly + weather | có | có (`data_quality.md`) |
| E01 | có | MAE 0.255 kWh |
| E02 | có | MAE 0.230 kWh (horizon **1h**, không so trực tiếp E01) |
| LSTM 2×2 | có | E03 0.235; E03b 0.237; E04/E04b 0.225 (3 seed) |
| PatchTST E05 | có | mean MAE 0.219 (3 seed) |
| Pearson builder | có | heatmap EDA; mean edge 0.0575 |
| GCGRU E06/E07 | có | E06 0.231; E07 0.230 (kém E04b → RQ1 âm) |
| Few-shot E08 | có (`personalize.py`) | 3 seed: P1@K=90 mean MAE **0.222**; P2@K=1 mean 0.226 |

## Test hiện có kiểm tra gì

Leak graph, mask metric, House 14, zero-hour missing, sample-count, scaler bỏ test, LSTM/PatchTST/GCGRU/personalize smoke. Test **không** thay số REFIT trong luận văn.

## Nên đọc tiếp trong repo (ngoài thư mục này)

- `DATA_LEAKAGE.md` — luật
- `thesis/research_plan.md` — RQ và chapter map
- `reports/tong-hop-tu-dau.md` — tiến độ + số
- `reports/nguyen-lieu-mail-thay.md` — nguyên liệu mail thầy
- `reports/literature_review.md` — paper đã có
- `reports/data_quality.md` — số đo trên file Kaggle
- `src/data/pipeline.py` — đọc song song với `03-du-lieu-refit.md`
