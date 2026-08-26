# Feature và scaler

## `src/data/feature_sets.py`

`resolve_columns(weather_set, available)` trả về `(history_cols, future_cols)`.

- History luôn bắt đầu bằng `kwh_input` (load đã fill ngắn).
- Future không chứa load.
- Weather cột nào không có trên panel thì bỏ im lặng (demo thiếu cột vẫn chạy).

Không nhét appliance vào đây.

## Hai scaler — `src/data/scalers.py`

### `HouseLoadScaler`

Mean/std **từng nhà**, chỉ giờ `split==train` và `observed`. Nhà tiêu thụ 0.3 kWh/h và nhà 1.2 kWh/h không nên dùng chung một scale nếu so individual; với global, scale theo nhà vẫn công bằng hơn (model học hình dạng, embedding học mức).

`inverse()` lúc đánh giá: MAE phải ra kWh. Báo cáo số scaled là sai khoa học (không so được paper, không so được hai nhà).

### `ColumnScaler`

Weather **chung cả thị trấn**, vẫn fit train only. Không scale sin/cos calendar (đã ~[-1, 1]).

`_scale_panel` trong `lstm_run.py` phân biệt weather vs calendar bằng tiền tố tên cột (`hour_`, `dow_`, `month_`, `is_`). Cách này hơi thô nhưng khớp đúng bộ cột hiện tại.

## Hai cột load cạnh nhau

| Cột | Ai dùng | Có impute? |
|-----|---------|------------|
| `kwh` | target, metric | Không |
| `kwh_input` | LSTM history | Có, tối đa 3 giờ |
| `observed` | mask loss + metric | Không bao giờ lật True sau impute |

Nhầm hai cột này là leak kiểu "học đáp án đã nội suy".
