# Split, cửa sổ, chống leak

File luật: `DATA_LEAKAGE.md`. Module: `src/data/splits.py`, `src/data/windowing.py`, `src/data/dataset.py`.

## Split thời gian, không phải shuffle hàng

```text
|-------- train 70% --------|--- val 15% ---|---- test 15% ----|
```

Cắt trên **timeline chung** (mọi nhà cùng mốc). Shuffle row = nhà "nhìn" tương lai của chính nó và của hàng xóm.

Trên REFIT đã build:

- start: 2014-03-20 12:00 UTC
- train_end: 2015-01-06
- val_end: 2015-03-09
- test_end: 2015-05-10

Val chỉ để early stopping / chọn epoch. Test chạm một lần sau khi đóng băng protocol.

## Cửa sổ 168 / 24

`origin` = giờ **đầu tiên** của horizon (giờ sắp forecast).

```text
[---- lookback 168h ----][---- horizon 24h ----]
                         ^ origin
```

`drop_crossing_windows()`:

- train: cả lookback và horizon nằm trong `[start, train_end]`
- val: origin sau `train_end`, horizon không vượt `val_end`
- test: origin sau `val_end`, horizon không vượt `test_end`

Cửa sổ đè lên biên bị **bỏ**, không cắt horizon cho khớp. Cắt horizon = đổi bài toán.

## `build_windows` — vì sao cắt bằng chỉ số hàng

Panel đã dense hourly. `iloc[i-168:i]` và `iloc[i:i+24]` tương đương `date_range` nhưng nhanh và không lệch DST. Bỏ cửa sổ nếu history hoặc future covariate có NaN (`kwh_input` chưa fill hết, hoặc weather thiếu).

Target `y` lấy từ `kwh` (có thể NaN). `mask` lấy từ `observed`. Tensor `y` thay NaN bằng 0 **chỉ để nhân loss**; mask = 0 thì giờ đó không vào MSE.

## Fit chỉ trên train

Không được ước lượng trên full series "vì N nhỏ":

- `HouseLoadScaler`, `ColumnScaler` (weather)
- Pearson graph
- Mọi thống kê đa nhà (kể cả scale của MASE: in-sample **train**)

Test leak trong `tests/test_leakage.py`: đầu độc test bằng spike khổng lồ. Graph train-only không đổi; graph fit all-split thì correlation sập về ~1.

## Personalization (chưa implement, đã chốt protocol)

```text
|-------- train --------|--- val ---|-- adapt K --|---- test ----|
                                     K ngày ngay trước test
```

K ∈ {1, 7, 30, 90}. Adapt **không** chồng test. Early stopping P1/P2 = 20% cuối cửa sổ adapt, không phải test.

P0 đóng băng global. P1 chỉ head + household embedding. P2 full, LR = 1/10.

K=1 ngày = 24 **target**. Lookback 168h lấy context từ trước cửa sổ adapt (val/cuối train). Hợp lệ vì là quá khứ của target. Đưa giờ test vào input hoặc early stop = leak.

Hàm `adaptation_window()` đã có trong `splits.py`. Vòng fine-tune chưa viết (`12-chu-chua-viet.md`).

## E10

4 nhà hold-out (placeholder trong `data.yaml`: 2, 8, 16, 20 — chọn lại cho đủ loại occupancy). Graph xây lại trên nhà còn lại + train timestamps. Số E10 **không** so trực tiếp bảng chính (N train ~13).
