# Tại sao không làm cách khác

Các lựa chọn đã cân, không phải mặc định thư viện.

## Hourly, không 8 giây / 30 phút

8 giây là thang NILM. ST-GNN + weather ERA5 hourly. Meyer et al. cũng hourly REFIT. Kaggle GPU 12h không đủ raw 8s. 30 phút chỉ là sensitivity nếu hourly che đỉnh tối.

## 24h horizon, không 1h/6h/48h

Supervisor so GNN vs LSTM trên STLF ngày tới. Thêm horizon = nhân cả matrix. Persistence 1h chỉ để calibrator kỳ vọng.

## Lookback 168h

Một tuần: chu kỳ ngày + cuối tuần. Ablation 24/72/168 chỉ trên model thắng, không phải mọi model.

## Pearson cố định, không AGCRN làm model chính

N=17. Adaptive adjacency quá nhiều tự do so với chứng cứ (Nguyen 2025). AGCRN = ablation overfit.

## Không graph địa lý

REFIT ẩn vị trí nhà. Bịa tọa độ = bịa cạnh.

## LSTM trước Transformer/TSFM

RQ1 cần GNN vs LSTM, cùng điều kiện. PatchTST/TSFM là mốc "model hiện đại", không phải hypothesis chính.

## Individual × Global tách weather

Đổi hai yếu tố một lúc không ước lượng được hiệu ứng hàng xóm.

## Loại PV 3/11/21 khỏi bảng chính

Aggregate nhiễm phát điện. Robustness subset riêng, không trộn.

## Mask vs fill zero

Fill gap bằng 0 là artifact cleaning. Metric trên imputed = tự lừa.

## ERA5 oracle, không giả vờ forecast 2014

Dữ liệu forecast vận hành không tồn tại cho 2013–2015 trên API đó.

## Không MAPE, không metric scaled

Zero; và scaled MAE không so được giữa nhà / giữa paper.

## Không London để "làm GNN đẹp"

London nằm pretraining Chronos/Moirai. REFIT là out-of-sample sạch hơn. N nhỏ mà graph không giúp vẫn là kết quả hợp lệ (Plan B).
