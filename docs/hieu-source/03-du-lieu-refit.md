# Dữ liệu REFIT: load, gom giờ, chất lượng

## File thô trên Kaggle

Upload `kyleahmurphy/uk-electrical-load` đặt tên `House_1.csv` … `House_21.csv`. Bản Zenodo/paper thường là `CLEAN_House1.csv`. Loader nhận **cả hai**.

Cột thật trong file Kaggle:

| Cột | Ý nghĩa |
|-----|---------|
| `Time` | Đồng hồ tường UK, đã chỉnh BST |
| `Unix` | Giây UTC (ReadMe viết "UCT") |
| `Aggregate` | Công suất cả nhà, Watt |
| `Appliance1`–`9` | Không dùng cho forecast |

**Không có House 14.** Code không được viết `range(1, 21)`. `discover_house_files()` đọc directory, parse số từ tên file. Nếu xuất hiện House 14 thì raise.

## `src/data/loading.py` — làm gì

- Tìm file, map `house_id`.
- Chỉ đọc cột thời gian + `Aggregate` (`usecols`) để khỏi kéo 9 cột appliance (~tiết kiệm RAM).
- `iter_raw_houses()` yield **từng nhà**. File ~350 MB × 17 nhà không được nằm cùng lúc trong RAM.
- Loại House 3, 11, 21 ở bước iterate (cấu hình `exclusions.solar_contaminated`). House 1, 6, 7 được rewire PV, **giữ**.

### Vì sao loại 3 / 11 / 21

Cảm biến kẹp dòng không biết chiều công suất. PV ban ngày hiện như **tải tăng** hình chuông. `Aggregate` không còn là tiêu thụ. Để trong main table sẽ làm GNN học "nhà nắng thì watt cao" — đó không phải STLF.

## `src/data/aggregate.py` — 8 giây thành hourly kWh

Công thức:

```text
hourly_kwh = mean(watts) / 1000
```

Trung bình Watt trong một giờ, chia 1000, đúng bằng kWh của giờ đó. Không dùng `sum(W) * 8s` vì số sample mỗi giờ không đều.

### Guard số sample

Raw ~8 giây → khoảng 450 điểm/giờ. `min_samples_per_hour: 90`. Giờ chỉ còn vài điểm (mất kết nối) bị đánh **missing**, không được coi là mean thật.

### Zero giả

Bản cleaned fill gap > 2 phút bằng **0 Watt**. Tủ lạnh thật không cho cả giờ = 0. Giờ mean đúng 0 W và đủ sample → missing. Đây là lý do không lead bằng MAPE: zero phá MAPE và một phần zero là outage đã fill.

### Thời gian: dùng Unix, không localize `Time`

`Time` là wall-clock. Ngày 27/10/2013 (lùi giờ) có giờ lặp → `tz_localize(..., ambiguous="infer")` vỡ. Unix đã là UTC. `watts_to_hourly_kwh` ưu tiên Unix. Nhánh DateTime chỉ còn cho test/fixture không có Unix, và dùng `ambiguous="NaT"` (bỏ giờ mơ hồ).

Cột ra: `household_id`, `timestamp` (UTC, floor giờ), `kwh`, `observed`, `n_samples`.

## `src/data/quality.py` — ma trận N × T

GNN cần cùng một lưới thời gian cho mọi node. REFIT: nhà bắt đầu/kết khác nhau, uptime paper ~88%.

1. **Common window.** Giao của `[first, last]` observed từng nhà. Nếu giao < 12 tháng: loại nhà làm hẹp cửa sổ nhất, ghi vào báo cáo. Thực tế đã đo: 2014-03-20 → 2015-05-10, **13.7 tháng**, không phải loại thêm nhà.
2. **Coverage ≥ 80%** trong cửa sổ. 17 nhà đều > 0.99 trong cửa sổ chung (đuôi mất mát đã bị cắt).
3. **`dense_panel`.** Reindex mọi nhà lên cùng `date_range` hourly UTC. Giờ không có hàng → `observed=False`.
4. **`impute_short_gaps`.** Nội suy tuyến tính tối đa 3 giờ vào cột **`kwh_input`**. Cột `kwh` và `observed` **không đổi**.

### Vì sao impute input nhưng không impute target

LSTM cần 168 bước liên tục. Thiếu 1–2 giờ giữa tuần sẽ vứt cả cửa sổ. Nội suy ngắn cho input là chấp nhận được. Nếu tính MAE trên giá trị đã nội suy, model được thưởng vì "đoán đúng chỗ mình bịa". Mọi metric chỉ nơi `observed=True`.

## `src/data/metadata.py`

Table 2 Murray et al. (2017): số người, tuổi nhà, số thiết bị, loại nhà, số phòng ngủ. Kaggle **không** kèm bảng này nên transcript + cite trong code. Dùng cho graph metadata và feature tĩnh. House 2 không có năm xây → NaN, scaler/graph sẽ median-fill.

## `src/data/demo.py`

Panel giả (6 nhà, 180 ngày) khi chưa có REFIT. Cùng schema với parquet thật. Chỉ để test code. Số trên demo **không** đưa vào thesis.

## Kết quả đã ghi (file thật)

Xem `reports/data_quality.md`. Tóm tắt: loại 3/11/21, giữ 17 nhà, House 20 cắt đầu cửa sổ, House 8 cắt đuôi.
