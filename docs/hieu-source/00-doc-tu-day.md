# Đọc source từ đây

Thư mục này giải thích **toàn bộ code hiện có**: mỗi phần làm gì, vì sao viết như vậy, và nó trả lời câu hỏi nghiên cứu nào. Không phải hướng dẫn chạy experiment.

Đọc theo thứ tự số. File sau giả định bạn đã hiểu file trước.

| File | Nội dung |
|------|----------|
| [00-doc-tu-day.md](00-doc-tu-day.md) | Mục lục này |
| [01-bai-toan-va-gia-thuyet.md](01-bai-toan-va-gia-thuyet.md) | Thesis hỏi gì, code phải chứng minh gì |
| [02-kien-truc-tong.md](02-kien-truc-tong.md) | Luồng dữ liệu end-to-end, map thư mục |
| [03-du-lieu-refit.md](03-du-lieu-refit.md) | Loader, aggregation, quality, mask |
| [04-thoi-tiet-va-lich.md](04-thoi-tiet-va-lich.md) | Open-Meteo, UTC, calendar |
| [05-split-window-leakage.md](05-split-window-leakage.md) | Cắt thời gian, cửa sổ 168/24, cấm gì |
| [06-feature-va-scaler.md](06-feature-va-scaler.md) | Cột nào vào model, scale ở đâu |
| [07-mo-hinh.md](07-mo-hinh.md) | Naive, LSTM, stub PatchTST/GCGRU |
| [08-do-thi.md](08-do-thi.md) | Pearson, metadata, adaptive |
| [09-train-va-danh-gia.md](09-train-va-danh-gia.md) | Loss, early stopping, metrics, log |
| [10-config-va-experiment.md](10-config-va-experiment.md) | YAML, scripts, notebook |
| [11-tai-sao-khong-lam-cach-khac.md](11-tai-sao-khong-lam-cach-khac.md) | Các lựa chọn đã loại |
| [12-chu-chua-viet.md](12-chu-chua-viet.md) | Stub, việc còn lại |

Quy ước trong tài liệu:

- Tên file / hàm / cột giữ nguyên tiếng Anh, vì đó là tên trong code.
- "Làm gì" = trách nhiệm của module.
- "Vì sao" = lý do khoa học hoặc kỹ thuật, không phải "vì tiện".
- File gốc của protocol chống leak: [`DATA_LEAKAGE.md`](../../DATA_LEAKAGE.md). Nếu code và file đó lệch nhau thì sửa code.
