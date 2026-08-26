# Train và đánh giá

## Seed — `src/train/seeds.py`

Một hàm `set_seed`: Python, NumPy, `PYTHONHASHSEED`, PyTorch CPU/CUDA, `cudnn.deterministic`. YAML khai `seeds: [0, 1, 2]` cho model có học. Naive chạy một lần. N=17, khác biệt 1-seed không đáng tin (Nguyen et al. báo std).

## Vòng train — `src/train/trainer.py`

- Loss: MSE **nhân mask**. Giờ không observed không kéo gradient.
- Clip grad 1.0.
- Early stopping: MAE val **đã inverse về kWh**, patience từ YAML. Không stop trên loss scaled (scale khác nhà → số không đọc được).
- Trả best weights theo val MAE thấp nhất.
- `num_workers=0`: Windows + reproducibility.

`evaluate_kwh`: predict → inverse scaler theo `household_id` từng bước horizon → `masked_metrics`.

## Metrics — `src/eval/metrics.py`

Luôn mask + finite. Đơn vị kWh.

| Metric | Vai trò |
|--------|---------|
| MAE | Chính, đưa vào Wilcoxon |
| RMSE | Phạt lệch lớn (đỉnh tối) |
| sMAPE | % đối xứng, chịu zero tốt hơn MAPE |
| WAPE | Tổng sai / tổng tải — ổn khi nhà khác scale |
| MASE | MAE / MAE seasonal-naive **in-sample train**, chu kỳ 24h |

**Không lead MAPE.** Zero (thật hoặc fill) làm MAPE nổ.

Wilcoxon: `src/eval/stats.py`, vector MAE **từng nhà**. N=17 power thấp → luôn báo median diff, số nhà thắng, không chỉ p-value. Diebold–Mariano chỉ cặp cuối, trên forecast origin; chưa implement.

## Experiment log — `src/train/experiment_log.py`

Append-only CSV. Cột: experiment id, seed, hash config, git commit, metrics, thời gian, số tham số. **Không sửa tay.** Hai dòng E01 trong log hiện tại: dòng 6 nhà = demo; dòng 17 nhà = REFIT thật (MAE 0.255).

## Điểm vào — `scripts/run_experiment.py`

Đọc YAML → load parquet → naive hoặc `run_lstm`. Model khác → `NotImplementedError`. `--max-epochs` để smoke, không đụng YAML gốc.

## Plot EDA — `src/eval/plots.py`

Missing heatmap, profile giờ, Pearson train-only, scatter load vs nhiệt (bằng chứng RQ3). Gọi từ `scripts/run_eda.py`.
