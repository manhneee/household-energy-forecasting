# Bài toán và giả thuyết

## Thesis này không làm gì

Không tối ưu hóa hóa đơn. Không dự báo từng appliance. Không invent một GNN mới. Không claim "GNN + weather + hộ gia đình" là mới — tổ hợp đó đã có paper 2025–2026.

Code vì thế **không** có module giá điện, không lấy `Appliance1`–`Appliance9` làm node, không có hybrid GNN–foundation.

## Thesis hỏi ba câu đo được

1. **RQ1 (supervisor).** Cùng weather, cùng split, cùng horizon 24 giờ: ST-GNN có hơn LSTM không, khi 17 nhà chung một thị trấn và **một** chuỗi thời tiết?
2. **RQ2 (đóng góp).** Cần bao nhiêu lịch sử nhà đích (1 / 7 / 30 / 90 ngày) để thích nghi model global? Graph giúp hay hại bước thích nghi đó?
3. **RQ3 (ablation).** Weather set nào (không / chỉ nhiệt / bộ chọn) và graph nào (không / Pearson train-only / metadata / adaptive) đổi độ chính xác, **sau khi** đã chặn leak thời gian?

Mọi file trong `src/` tồn tại để trả lời một trong ba câu này, hoặc để bảo vệ tính trung thực của câu trả lời (mask, scaler train-only, UTC).

## Task mà code đang giải

- Đầu vào: lịch sử tải **168 giờ** của một nhà, cộng calendar, cộng weather (tùy experiment).
- Đầu ra: 24 số kWh tiếp theo của **cùng nhà đó** (`Aggregate`, đã quy đổi hourly).
- Đơn vị báo cáo: **kWh**, không phải số đã scale.
- Tần số: hourly. Dữ liệu thô 8 giây chỉ dùng để tính trung bình giờ.

## Bốn cách "nhìn" hộ gia đình

Đây là trục so sánh chính, không phải zoo model.

| Cách | Ý nghĩa | Experiment |
|------|---------|------------|
| Individual | Một model / một nhà. Không học từ nhà khác. | E03, E03b |
| Global | Một model cho mọi nhà. Household embedding phân biệt nhà. | E04, E04b, E05 |
| Graph | Nhà là node; cạnh = tương quan hoặc metadata. | E06, E07 |
| Personalized | Lấy global/graph rồi thích nghi ít ngày trên nhà đích. | E08 |

LSTM 2×2 (individual/global × không weather/có weather) tách **hai yếu tố**. So "individual không weather" với "global có weather" sẽ không biết cái nào giúp.

## Dataset khóa

REFIT, Loughborough, 2013–2015, upload Kaggle `kyleahmurphy/uk-electrical-load`. Cùng thị trấn nên **một điểm Open-Meteo** là đủ. N=20 (thiếu House 14). Main table loại House 3, 11, 21 vì PV dính vào `Aggregate`.

Kết quả đã đo trên file thật: 17 nhà, cửa sổ chung 13.7 tháng. Chi tiết trong `reports/data_quality.md`.
