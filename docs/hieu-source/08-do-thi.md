# Đồ thị giữa các hộ

GNN **chưa train**. Ba builder đã có để RQ3 không bị bịa graph lúc viết model.

## Bốn họ graph trong literature — REFIT cho phép cái nào

| Họ | Trên REFIT | Module |
|----|------------|--------|
| Địa lý / feeder | **Không thể.** Paper không công bố tọa độ/địa chỉ. | Nói rõ Chương 4, đừng để trống |
| Thống kê | Pearson kNN, **train hours only** | `src/graphs/correlation.py` |
| Metadata / hybrid | Table 2: occupancy, năm, thiết bị, loại nhà, phòng ngủ | `src/graphs/metadata.py` |
| Học (adaptive) | AGCRN; dễ overfit N≈17 | `src/graphs/adaptive.py` (stub, raise) |

Nguyen et al. (2025) trên London: cách dựng graph gần như không đổi kết quả; graph học overfit. Thesis này **lặp lại câu hỏi đó trên REFIT**, có thêm weather.

## Pearson kNN — vì sao train-only

Tương quan cả chuỗi (kể cả test) = nhà "nhìn" đồng biến tương lai. `pearson_knn()` pivot `split==train`, k=3 theo `|corr|`, cạnh âm bị cắt (`max(score, 0)`), đối xứng hóa, rồi chuẩn hóa `D^{-1/2}(A+I)D^{-1/2}` (Chuẩn GCN).

k=3: N=17, k lớn → gần như fully connected, hết ý nghĩa "hàng xóm".

## Metadata similarity

Cosine trên feature đã StandardScaler. House 2 thiếu năm xây → median. Cũng kNN k=3, cùng hàm chuẩn hóa. Đây là graph **không** học từ load — đối trọng với Pearson (Pearson có thể chỉ copy "nhà giống nhau vì cùng giờ nấu cơm").

## Adaptive

Chưa viết. Giữ làm Ablation D, không phải model chính.

## Node không phải appliance

Paper IEEE TAI đã làm GNN appliance **trong một nhà**. Câu hỏi khác. Dùng lại là trùng đề.
