# Báo cáo tiến độ tuần

**Kính gửi:** Giáo viên hướng dẫn  
**Từ:** Luận văn dự báo phụ tải hộ gia đình  
**Tuần:** 17–24 tháng 8 năm 2026  
**Phiên bản repo:** 0.2.0 (22 tháng 8 năm 2026)  
**Tên đề tài (working title):** Few-Shot Personalization of Weather-Aware Spatio-Temporal Models for Neighborhood-Scale Household Electricity Forecasting

Bản tiếng Anh: [`weekly_report_2026-08-24.md`](weekly_report_2026-08-24.md)

---

## 1. Tình trạng trong một đoạn

Tuần này đề tài đã chuyển từ research plan sang **pipeline chạy được, chống rò thông tin (leakage-controlled) trên file REFIT thật**. Panel hourly, ghép thời tiết, protocol chất lượng dữ liệu, và baseline seasonal-naive đã được đo trên 17 hộ Loughborough. LSTM có thời tiết đã được viết code và smoke-test trên panel giả 6 nhà; **chưa train trên REFIT**. Em không claim xếp hạng mô hình. Các số dưới đây là sự thật về dữ liệu và **một** baseline naive.

---

## 2. Việc đã làm xong

| Hạng mục | Tuần này |
|----------|----------|
| Phạm vi | Khóa 3 câu hỏi nghiên cứu (RQ), ma trận thí nghiệm E01–E10, và protocol chống leak (`DATA_LEAKAGE.md`). Ngôn ngữ claim: so sánh có kiểm soát, **không** invent GNN mới. |
| Dữ liệu | Loader tự tìm `CLEAN_House*.csv` (không có House 14). Watts 8 giây → kWh hourly, có ngưỡng số mẫu. Giờ đúng bằng 0 coi là missing (bản cleaned REFIT điền 0 vào khoảng trống). |
| Chất lượng | Protocol cửa sổ chung tính từ file Kaggle, không lấy số trung bình của paper 2017. Báo cáo chất lượng ghi từ file thật. |
| Thời tiết | Open-Meteo ERA5 cho Loughborough (52.77°N, 1.21°W): nhiệt độ, bức xạ sóng ngắn, độ ẩm. REFIT gắn `Europe/London` rồi đổi sang UTC trước khi merge. Kiểm tra đỉnh nắng buổi trưa: đạt. |
| Split | Cắt thời gian 70 / 15 / 15 theo timeline chung. Scaler và graph chỉ fit trên train. |
| Mô hình | Seasonal naive (E01) chạy trên REFIT. Persistence (E02) có code, chưa ghi log. LSTM Individual/Global × có/không thời tiết đã viết; chỉ smoke trên panel demo. |
| Đồ thị | Pearson kNN (k=3) train-only đã có. Metadata chép từ Murray et al. (2017) Table 2. Adaptive graph và GCGRU/PatchTST vẫn là stub. |
| Test | Leak, metric có mask, House 14, giờ 0 = missing, sample-count, scaler không đụng test, LSTM smoke (2 epoch trên demo). |
| Đọc paper | Log tài liệu: ST-GNN phụ tải hộ, weather + deep model, TSFM trên REFIT hourly, personalization. Lập trường: tổ hợp GNN + thời tiết + hộ gia đình **đã có paper**. |

---

## 3. Phân tích dữ liệu (phần em sẽ trình trong buổi họp)

### 3.1 Cohort sau khi loại nhà

- **Tìm thấy:** 20 nhà (ID 1–13, 15–21). Không có House 14.
- **Loại khỏi bảng chính:** House 3, 11, 21 (PV dính vào `Aggregate`). House 1, 6, 7 đã rewire và **giữ lại**.
- **Giữ:** 17 nhà. Không nhà nào bị loại vì coverage hay vì làm ngắn cửa sổ chung.
- **Cửa sổ chung (UTC):** 2014-03-20 12:00 → 2015-05-10 23:00 = **9.996 giờ (13,7 tháng)**.
- **Coverage trong cửa sổ đó:** 0,993–0,999 (trung bình ≈ 0,997). Thấp nhất: House 13.
- **Tỷ lệ missing** sau aggregation, trước dense reindex: **1,3%**.
- Coverage trên span riêng của từng nhà kém hơn (House 2: 0,758; House 13: 0,783). Training dùng **giao** các span, không dùng số “88% mean uptime” của paper 2017.

**Hệ quả:** N = 17 nhỏ cho GNN. Kết quả graph âm (GNN không thắng) vẫn là kết quả Bachelor hợp lệ — đã ghi trong research plan. Cửa sổ đủ dài cho lookback 168 giờ / horizon 24 giờ và khoảng test khoảng hai tháng.

### 3.2 Cắt thời gian (không xáo hàng)

| Split | Mốc kết thúc (UTC) | Vai trò |
|-------|--------------------|---------|
| Train (70%) | 2015-01-06 00:00 | Fit scaler, graph, model |
| Val (15%) | 2015-03-09 11:00 | Early stopping / chọn mô hình thôi |
| Test (15%) | 2015-05-10 23:00 | Đánh giá khi đã đóng băng |

Cửa sổ nào có horizon 24 giờ cắt qua ranh giới split thì bị loại. Thời tiết trên horizon lúc test là **ERA5 oracle** (Open-Meteo Historical Forecast không phủ 2014). Chương 7 phải nói rõ: đây **không** phải setup dự báo ngày-tới vận hành.

Kiểm tra căn giờ sau merge: giờ đỉnh bức xạ sóng ngắn trung bình = **12,99** (kỳ vọng 12, dung sai ±2). **Đạt.** Lệch 1 giờ mùa hè sẽ là bug UTC im lặng.

### 3.3 Heterogeneity phụ tải (chỉ giờ `observed`)

Mức tiêu thụ hourly trung bình trên panel khoảng **0,49 kWh/h**, nhưng các nhà **không** thay thế được nhau:

| Hộ | Số người | Mean kWh/h | Median kWh/h | Max kWh/h |
|--:|--:|--:|--:|--:|
| 15 | 1 | 0,257 | 0,187 | 4,04 |
| 19 | 4 | 0,293 | 0,220 | 2,00 |
| 4 | 2 | 0,366 | 0,322 | 3,49 |
| 12 | 3 | 0,374 | 0,227 | 4,54 |
| 20 | 2 | 0,377 | 0,287 | 2,43 |
| 17 | 3 | 0,413 | 0,247 | 4,63 |
| 1 | 2 | 0,439 | 0,280 | 12,51 |
| 18 | 2 | 0,449 | 0,361 | 4,96 |
| 2 | 4 | 0,470 | 0,211 | 5,10 |
| 6 | 2 | 0,470 | 0,406 | 2,96 |
| 7 | 4 | 0,533 | 0,231 | 4,41 |
| 13 | 4 | 0,565 | 0,341 | 4,46 |
| 16 | 6 | 0,569 | 0,436 | 6,01 |
| 9 | 2 | 0,584 | 0,255 | 5,82 |
| 8 | 2 | 0,665 | 0,426 | 6,11 |
| 10 | 4 | 0,709 | 0,435 | 5,08 |
| 5 | 4 | 0,759 | 0,581 | 8,37 |

**Ba điểm em sẽ bảo vệ:**

1. **Số người ở là proxy yếu.** House 15 (1 người) mean thấp nhất, đúng kỳ vọng; nhưng House 19 (4 người) gần như thấp tương đương, còn House 8 (2 người) nằm nhóm cao nhất. Model global không có household embedding (hoặc graph) sẽ hòa các mức này vào một.
2. **Lệch phải / spike.** Median << mean ở nhiều nhà (House 7: 0,23 vs 0,53; House 1 max 12,5 kWh). RMSE sẽ xấu hơn MAE. Vì thế protocol dẫn bằng MAE / WAPE / MASE, không dẫn MAPE.
3. **Đó là lý do có RQ2.** GNN hàng xóm trên N = 17 dễ overfit vài nhà tải cao. Few-shot personalization (1 / 7 / 30 / 90 ngày) là phần đóng góp không cần claim kiến trúc mới.

### 3.4 Baseline trên REFIT (chỉ E01)

Seasonal naive = copy 24 giờ của ngày hôm trước. Metric chỉ trên **giờ test đã quan sát**, đơn vị kWh.

| Thí nghiệm | Dữ liệu | N nhà | MAE | RMSE | sMAPE | WAPE | MASE |
|------------|---------|------:|----:|-----:|------:|-----:|-----:|
| E01 seasonal naive | REFIT, cửa sổ chung | 17 | **0,255** | 0,500 | 39,8 | 0,537 | 1,007 |
| E01 (sanity) | Demo giả 6 nhà | 6 | 0,051 | 0,064 | 11,6 | 0,080 | 0,696 |
| E04b LSTM smoke | Demo giả 6 nhà | 6 | 0,037 | 0,047 | 8,5 | 0,058 | 0,507 |

**Đọc số E01 trên REFIT:**

- MAE **0,255 kWh** so với mean cohort ~0,49 kWh tức **WAPE 54%**. Profile hôm qua là baseline nghiêm của phụ tải nhà ở theo chu kỳ ngày; model học được phải thắng trên cùng cửa sổ test đã mask.
- MASE ≈ **1,00** là đúng kỳ vọng: thang seasonal in-sample và lỗi seasonal-naive trên test cùng một họ dự báo.
- RMSE ≈ 2× MAE khớp đuôi spike đã thấy ở max từng nhà.
- Số LSTM **0,037 không so được** với 0,255. Đó là smoke ~2 epoch trên data giả (16.336 tham số, ~17 giây). Em không đưa số này vào bảng kết quả luận văn.

E02 (persistence, chẩn đoán 1 giờ) đã có code, chưa log trên REFIT.

---

## 4. Quyết định phương pháp cần thầy/cô xác nhận (đã nằm trong code)

Đây không phải kết quả. Đây là quyết định nếu không chốt thì bảng sau này không đọc được.

1. **Một chuỗi thời tiết cho cả thị trấn.** 17 nhà dùng chung ERA5 Loughborough. RQ1 đúng là: ST-GNN còn giúp không khi **không** có biến thiên thời tiết theo không gian?
2. **Thời tiết trên horizon là ERA5 oracle.** Trung thực với 2013–2015; không phải forecast vận hành năm 2014.
3. **Metric không thấy target đã nội suy.** Khoảng trống ngắn (≤ 3 giờ) có thể nội suy cho *đầu vào* model; giờ `observed = False` không vào MAE/RMSE/sMAPE/WAPE/MASE.
4. **Lưới LSTM 2×2** (individual/global × không thời tiết/có thời tiết) trước mọi GNN, để không lẫn yếu tố thời tiết với pooling.
5. **Plan B / C** đã viết sẵn: bỏ GNN nếu không train được; luận văn tối thiểu là E01 + E04b + E07 + một ablation thời tiết + Wilcoxon trên 17 nhà.

---

## 5. Ma trận thí nghiệm — trạng thái thật

| ID | Mô hình | Thời tiết | Graph | Code | Số trên REFIT |
|----|---------|-----------|-------|------|----------------|
| E01 | Seasonal naive | không | không | có | MAE 0,255 kWh |
| E02 | Persistence | không | không | có | chưa log |
| E03 / E03b | LSTM individual | không / có | không | có | chưa |
| E04 / E04b | LSTM global | không / có | không | có | chỉ demo |
| E05 | PatchTST | có | không | stub | — |
| E06 / E07 | GCGRU / T-GCN | không / có | Pearson | stub | — |
| E08 | Few-shot P0/P1/P2 | có | theo nguồn | mới có protocol | — |
| E09 | Một TSFM | nếu hỗ trợ | không | chưa | — |
| E10 | Leave-4-out | có | Pearson | mới có list YAML | — |

Holdout E10 (chờ xác nhận): House **2, 8, 16, 20** (số người 4 / 2 / 6 / 2; tải lẫn). Chọn để bốn nhà không cùng một cỡ.

---

## 6. Việc đề xuất tuần sau

1. Chạy **E02** trên cùng panel REFIT, log cạnh E01.
2. Train **E04b** (LSTM global có thời tiết) trên REFIT, seed {0, 1, 2}, báo MAE từng nhà so với E01 (Wilcoxon khi có đủ 17 cặp).
3. Nếu còn compute: **E03b** để individual vs global không bị lẫn với thời tiết.
4. Xuất bốn hình EDA (missing heatmap, daily profile, Pearson train, load vs nhiệt độ) — code vẽ đã có, file PNG chưa có trong repo.
5. **Không** bắt đầu GCGRU trước khi lưới LSTM 2×2 có số trên REFIT. Không thì không gán được phần thắng cho yếu tố nào.

---

## 7. Câu hỏi xin ý kiến thầy/cô

1. Thời tiết horizon **ERA5-oracle** có chấp nhận được nếu luận văn ghi rõ, hay nên ablation “không thời tiết trên horizon” làm proxy vận hành?
2. Với N = 17, Plan A (GNN + few-shot) vẫn là đường chính, hay em nên lấy **Plan B** (LSTM/PatchTST + personalization, không GNN) làm trục và để E07 ở phụ lục?
3. Xin xác nhận bộ holdout **E10** {2, 8, 16, 20} nay đã biết coverage, hoặc chỉ định bốn nhà khác.
4. Kết quả graph **âm** (GNN không thắng LSTM có thời tiết) có được coi là câu trả lời RQ1 hợp lệ không, khi benchmark ST-GNN London đã xuất bản không dùng thời tiết?

---

## 8. Những điều em không claim

- LSTM thắng seasonal naive trên REFIT (chưa train).
- GNN + thời tiết + dự báo hộ là mới (literature đã có tổ hợp này; phần còn lại là so sánh có kiểm soát, vì paper London không đưa thời tiết).
- MAE 0,037 từ smoke demo là kết quả luận văn.
