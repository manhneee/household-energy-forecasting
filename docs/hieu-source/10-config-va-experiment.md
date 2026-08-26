# Config, experiment YAML, scripts

## Vì sao YAML chứ không phải argparse dài

Mỗi ô trong ma trận thí nghiệm = một file. So E04 và E04b = diff hai file (weather_set). Hash SHA256 của dict YAML ghi vào log — biết run nào đi với config nào.

## Hai lớp config

`configs/data.yaml` + `configs/weather.yaml`: **protocol** (cửa sổ, loại nhà, tọa độ). Đổi ở đây là đổi bài toán.

`configs/model/*.yaml`: kiến trúc (hidden, lr, epoch).

`configs/experiment/eXX_*.yaml`: một ô matrix. Field quan trọng:

| Field | Ý nghĩa |
|-------|---------|
| `id` | E01 … E10, khớp bảng thesis |
| `model` + `model_config` | Loại mạng |
| `weather_set` | `none` / `temperature` / `selected` |
| `graph` | `none` / `pearson_knn` / … |
| `mode` | `individual` / `global` / `graph` |
| `seeds` | list |

`src/config.py` `load_experiment()` merge model YAML vào `exp["model"]` và giữ `kind`.

## Ma trận Plan A

| ID | Đã có code? | Đã chạy REFIT? |
|----|-------------|----------------|
| E01 seasonal naive | có | có (MAE 0.255 kWh) |
| E02 persistence 1h | có | chưa |
| E03 LSTM individual, no wx | có | chưa |
| E03b LSTM individual, wx | có | chưa |
| E04 LSTM global, no wx | có | chưa |
| E04b LSTM global, wx | có | mới smoke demo |
| E05 PatchTST | stub | không |
| E06–E07 GCGRU | stub | không |
| E08 P0/P1/P2 | protocol only | không |
| E09 TSFM | chưa | không |
| E10 leave-4 | config placeholder | không |

## Scripts

| Script | Khi nào dùng |
|--------|----------------|
| `build_panel.py` | Một lần sau khi có `data/raw/House_*.csv` |
| `run_eda.py` | Sau panel, ra `reports/figures` |
| `run_experiment.py --config ...` | Từng ô YAML |
| `make_demo_panel.py` | Không dùng cho số thesis |

## Notebooks

`01_data_exploration.ipynb` gọi `build_hourly_panel` / load parquet. `02_baselines.ipynb` gọi `evaluate_naive`. Không chứa LSTM class.

## `src/paths.py`

Tìm raw: env `REFIT_RAW_DIR` → Kaggle `/kaggle/input/...` → `data/raw`. Nhận `House_*.csv` hoặc `CLEAN_House*.csv`, cả thư mục con (zip lồng).
