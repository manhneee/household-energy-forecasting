# Thời tiết và lịch

## Vì sao có weather

RQ1 và RQ3 hỏi weather có giúp STLF hộ gia đình khí **đốt gas, không HVAC** không. Kỳ vọng: nhiệt độ yếu hơn dataset sưởi điện; bức xạ / lịch vẫn có thể giúp (chiếu sáng, tủ lạnh, mùa). Ablation "không / chỉ nhiệt / bộ chọn" chính là câu trả lời, kể cả khi nhiệt không giúp.

Mọi nhà dùng **một** điểm Loughborough `(52.7721, -1.2052)`. Không có tọa độ từng nhà.

## `src/weather/open_meteo.py`

- API: Historical Weather `/v1/archive` (ERA5), docs trong `configs/weather.yaml`.
- Biến mặc định: `temperature_2m`, `shortwave_radiation`, `relative_humidity_2m`.
- Cache parquet + JSON metadata (lúc tải, tham số, ghi chú oracle).
- `merge_weather`: join theo `timestamp` UTC.
- `verify_solar_noon`: sau merge, đỉnh `shortwave_radiation` trung bình theo giờ **local** phải gần 12h ± 2h.

### Vì sao check noon

Open-Meteo mặc định UTC. REFIT `Time` là giờ London. Lệch 1 giờ mùa hè **không** hiện trong MAE nhưng phá hết ablation weather. Peak bức xạ ~13h local (đã đo 12.99) là bằng chứng merge đúng, không phải bằng chứng model giỏi.

Đã đo trên panel thật: `alignment.ok = true`, `mean_local_peak_hour ≈ 12.99`.

## Oracle future weather — phải nói trong luận văn

Horizon 24 giờ cũng dùng **cùng** ERA5. Đó không phải forecast năm 2014. API Historical Forecast của Open-Meteo chỉ có khoảng từ 2021–2022. REFIT là 2013–2015. Chương 7 phải viết rõ: kết quả là "nếu biết thời tiết tương lai hoàn hảo", upper bound, không phải vận hành.

`future_weather: era5_oracle` trong YAML tồn tại để khỏi quên.

## Calendar — `src/data/features.py`

Luôn gắn, không phải ablation weather:

- `hour` / `dow` / `month` dạng sin-cos (tránh 23 và 0 bị coi là xa nhau)
- `is_weekend`
- `is_uk_bank_holiday` — list cứng 2013–2015 (England & Wales), không gọi package `holidays` để Kaggle khỏi phụ thuộc mạng và list không đổi giữa các lần chạy

Calendar trên horizon là **biết trước** (không oracle). Khác bản chất với ERA5 tương lai.

## Ba weather set

Định nghĩa ở `configs/weather.yaml` và `src/data/feature_sets.py`:

| Set | Cột weather | Dùng khi |
|-----|-------------|----------|
| `none` | không | E03, E04, E06 — cô lập hiệu ứng graph/global |
| `temperature` | `temperature_2m` | Ablation RQ3 |
| `selected` | nhiệt + bức xạ + ẩm | E03b, E04b, E05, E07 |

History = `kwh_input` + calendar + weather (nếu có).
Future (horizon) = calendar + weather. **Không** có load tương lai — đó là thứ phải đoán.
