# Dữ liệu dùng chung cho bốn model

**Đơn vị phân tích:** một dòng = một khách hàng (`CustomerID`). Nguồn là `raw_data/Online Retail.xlsx`; chạy `scripts/run_pipeline.py` rồi `scripts/run_temporal_pipeline.py` để tạo các file bên dưới. Không cần melt bảng RFM.

| Giai đoạn | Feature để model đọc | Số khách | Cửa sổ kết quả 90 ngày |
| --- | --- | ---: | --- |
| Development | `data_cleaned/temporal/development/rfm_model_features.csv` | 2.716 | `[2011-06-01, 2011-08-30)` |
| Final | `data_cleaned/temporal/final/rfm_model_features.csv` | 3.314 | `[2011-09-01, 2011-11-30)` |

Mỗi file feature có `CustomerID`, `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled`. Chỉ **ba cột `_scaled`**, theo đúng thứ tự này, là ma trận `X`; `CustomerID` là khóa nối. Các cột được tạo bằng `log1p(RFM)` rồi `StandardScaler` fit trên giao dịch **trước** snapshot tương ứng. `rfm_customers.csv` cùng thư mục giữ `Recency` (ngày), `Frequency` (số hóa đơn duy nhất), `Monetary` (tổng mua hàng hợp lệ), `LastPurchase`, `LineCount`, `PrimaryCountry` để mô tả cụm. `manifest.json` ghi mốc thời gian và quy tắc dữ liệu.

```python
import pandas as pd

phase = "development"  # đổi thành "final" sau khi đã chốt cách làm
data = pd.read_csv(
    f"data_cleaned/temporal/{phase}/rfm_model_features.csv",
    dtype={"CustomerID": "string"},
)
X = data[["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]].to_numpy()
customer_ids = data["CustomerID"]
```

Chọn feature, tham số, seed và cách diễn giải bằng **development**. Sau đó chạy lại phương pháp đã chốt trên **final**, chỉ dùng feature quá khứ. Không đưa `CustomerID`, dữ liệu trong `data_cleaned/temporal/outcomes/`, hay các cột `repurchased_in_90d`, `future_invoices`, `future_revenue` vào model. Agglomerative và DBSCAN không có `predict` chuẩn; nếu lấy mẫu hoặc dùng quy tắc gán cụm, ghi rõ quy trình và phạm vi khách được đánh giá.

Xuất `models/outputs/development/<model>.csv` và `models/outputs/final/<model>.csv`, với **một dòng cho mỗi `CustomerID` của phase** và hai cột `CustomerID,cluster`. Tên `<model>` là `kmeans`, `agglomerative`, `dbscan`, `gmm`; DBSCAN dùng `-1` cho noise. Giữ nguyên ID, không loại khách hay xáo trộn tập khách. Có thể thêm `<model>.json` chứa `runtime_seconds` và `parameters`.

Khi đã có nhãn, chạy `python scripts/evaluate_temporal.py --phase development --with-models`. Chỉ chạy `--phase final --with-models` sau khi cả nhóm chốt model trên development. Script sẽ đối chiếu ID, tính chỉ số nội tại, hồ sơ cụm và so sánh lựa chọn 15% khách với quy tắc Monetary/RFM đơn giản.
