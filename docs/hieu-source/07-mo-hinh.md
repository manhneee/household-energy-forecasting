# Mô hình

## E01 / E02 — `src/models/naive.py` + `src/eval/baselines.py`

**Seasonal naive:** copy đúng 24 giờ của ngày hôm trước. Baseline STLF tối thiểu. Mọi deep model phải thắng cái này mới đáng kể.

**Persistence:** lặp giờ cuối. Chỉ diagnostic **horizon 1 giờ**. 1-step rất dễ; không train lại model cho 1 giờ. Có mặt để người đọc thấy 24h khó hơn 1h bao nhiêu.

Không có tham số, không seed.

## LSTM — `src/models/lstm.py`

Đã implement. Đây là đối thủ supervisor của GNN (cùng weather, cùng split, cùng 24h).

```text
x [B, 168, F]  --LSTM--> h_last [B, hidden]
hid [B]        --Embedding--> e [B, 8]
future [B, 24, C] --flatten--> f
concat(h_last, e, f) --MLP--> yhat [B, 24]
```

### Vì sao kiến trúc này

- LSTM trên lookback: đủ cho supervisor so GNN, không cần Transformer để trả lời RQ1.
- **Future covariates vào head, không vào LSTM.** LSTM không được thấy load tương lai; nó được thấy calendar + ERA5 oracle của 24 giờ tới. Đó là cách công bằng với "weather-aware".
- **Household embedding:** global model cần biết đang nói nhà nào. Embedding cũng là thứ P1 sẽ fine-tune (ít tham số, sống sót với K=1 ngày). `encoder_parameters()` / `head_parameters()` tách sẵn cho P1/P2.

### Individual vs global — `src/train/lstm_run.py`

| Mode | Việc code làm |
|------|----------------|
| `individual` | Lặp từng nhà: scaler riêng, model riêng (`n_households=1`), concat dự báo rồi tính metric chung |
| `global` | Một scaler-per-house (vẫn per house), một LSTM, embedding size = N |

Cùng class `LSTMForecaster`. Khác nhau ở dữ liệu và số embedding.

## PatchTST — `src/models/patchtst.py`

Stub. YAML E05 đã có. Implement sau khi LSTM 2×2 chạy trên REFIT. Vai trò: baseline Transformer **trained** (không phải TSFM). Có thể đổi iTransformer nếu weather không giúp PatchTST (benchmark lưới 2026).

## GCGRU / T-GCN — `src/models/gcgru.py`

Stub. Đây sẽ là model RQ1 phía GNN. Node = nhà, feature = load + weather chung. Graph = Pearson train-only. AGCRN adaptive chỉ ablation (N=17, graph học dễ overfit — Nguyen et al. 2025).

## TSFM (E09)

Chưa có file model. Ưu tiên Chronos-2 (có covariate + fine-tune mở). Fallback TiRex-2 zero-shot, rồi TTM. Meyer et al. đã làm TSFM univariate trên hourly REFIT — E09 chỉ là check có weather, không claim first-on-REFIT.

## Không viết

Informer, Autoformer, FEDformer, TimesNet, TimeMixer, N-BEATS, ASTGCN, DCRNN, GraphSAGE zoo, EnergyMamba. Lý do: nhân ma trận thí nghiệm, không trả lời RQ.
