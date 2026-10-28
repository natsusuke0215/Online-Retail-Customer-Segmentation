# Online Retail customer segmentation

Project dùng dữ liệu giao dịch Online Retail để tạo đặc trưng RFM và so sánh bốn thuật toán phân cụm. [Tài liệu project](docs/README.md) gồm kế hoạch, kết quả khảo sát và đánh giá phương pháp.

## Cấu trúc

```text
raw_data/                    Dữ liệu gốc, không chỉnh sửa
docs/                        Kế hoạch và đánh giá phương pháp
src/retail_segmentation/     Cleaning, RFM, EDA, đánh giá chung
scripts/                     Lệnh chạy pipeline và tổng hợp model
data_cleaned/                Dữ liệu chuẩn hóa, RFM, scaler, báo cáo chất lượng
reports/                     EDA, bảng/biểu đồ, bảng so sánh model
models/outputs/              Nhãn cụm do bốn thành viên xuất ra
tests/                       Kiểm tra logic trọng yếu
app.py                       Demo Streamlit đọc kết quả chung
```

## Chạy

Từ thư mục gốc project:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts/run_pipeline.py
.\.venv\Scripts\python scripts/run_temporal_pipeline.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m streamlit run app.py
```

Streamlit 1.57.0 cần Starlette tương thích; `requirements.txt` ghim cặp phiên bản đã kiểm tra để tránh lỗi import khi môi trường có Starlette cũ.

File dữ liệu gốc và output sinh ra không theo Git. Xem [hướng dẫn lấy dữ liệu](raw_data/README.md) để đặt `Online Retail.xlsx` đúng chỗ trước khi chạy pipeline. CI chỉ chạy test với dữ liệu giả lập nhỏ, không cần tải dữ liệu gốc.

Pipeline mặc định đọc `raw_data/Online Retail.xlsx`, giữ giao dịch trước snapshot `2011-12-10`. Có thể dùng `--source`, `--snapshot-date`, `--output-dir`, `--report-dir` để tạo một snapshot khác. **Dùng output dir khác** nếu chạy thử CSV hai năm để không ghi đè baseline.

Để kiểm định mua lại 90 ngày, dùng `scripts/run_temporal_pipeline.py` sau baseline. Teammate đọc `data_cleaned/temporal/development/rfm_model_features.csv` để chọn model, rồi tạo nhãn cho snapshot `final` với tham số đã chốt. Kết quả tương lai nằm riêng trong `data_cleaned/temporal/outcomes/`. Xem [hợp đồng bàn giao](docs/temporal_handoff.md); file `data_cleaned/rfm_model_features.csv` baseline cuối năm **không phù hợp** làm đầu vào cho kiểm định 90 ngày tại 01/09 vì chứa giao dịch tương lai.

## Hợp đồng dữ liệu cho bốn model

- Đọc `data_cleaned/rfm_model_features.csv`; lấy đúng ba cột `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled` theo thứ tự đó. `CustomerID` là khóa, không đưa vào model.
- `rfm_customers.csv` giữ RFM gốc và `PrimaryCountry` để mô tả cụm.
- Xuất một file CSV cho mỗi model vào `models/outputs/`: `kmeans.csv`, `agglomerative.csv`, `dbscan.csv`, `gmm.csv`. Mỗi file có **đúng** hai cột bắt buộc `CustomerID,cluster`, một dòng mỗi khách hàng; DBSCAN dùng `-1` cho nhiễu. Không xáo trộn/tự lọc khách hàng khi xuất nhãn.
- Có thể thêm `models/outputs/<model>.json` với `{"runtime_seconds": 1.23, "parameters": {"n_clusters": 4}}` để lưu thời gian chạy và tham số.
- Khi có nhãn, chạy `python scripts/evaluate_models.py` để tạo `reports/model_comparison.csv` và `reports/cluster_profiles.csv`. Ba chỉ số nội tại tính trên toàn bộ điểm được gán cụm; tỷ lệ nhiễu của DBSCAN luôn được báo cáo riêng. Silhouette và Calinski–Harabasz càng cao càng tốt, Davies–Bouldin càng thấp càng tốt; cần đọc cùng quy mô và ý nghĩa cụm.

## Quy tắc baseline

Giữ nguyên `raw_data/`. `transactions_clean.csv` chứa mọi dòng và cờ loại trừ; dữ liệu model chỉ gồm mua hàng có ID, ngày hợp lệ trước snapshot, số lượng và giá dương, hóa đơn không hủy, không trùng hoàn toàn, không thuộc mã phí/điều chỉnh `POST`, `C2`, `DOT`, `BANK CHARGES`, `M`, `23444` (Next Day Carriage). `Amount = Quantity × UnitPrice`; `Recency` tính bằng ngày, `Frequency` đếm hóa đơn duy nhất, `Monetary` là tổng `Amount`. Feature model là `log1p(RFM)` rồi `StandardScaler` fit trên snapshot baseline. Đây là một bài toán clustering không giám sát, nên không chia ngẫu nhiên dòng giao dịch thành train/test.

Các dòng giống hệt có thể là giao dịch thật do nguồn thiếu ID dòng hàng; `reports/eda_summary.md` và `data_cleaned/quality_report.json` ghi số lượng để kiểm tra độ nhạy trước khi diễn giải kết quả kinh doanh.

[Đánh giá phương pháp và nguồn tham khảo](docs/methodology_review.md) phân biệt những quy tắc có cơ sở từ UCI/scikit-learn với các giả định phân tích cần kiểm chứng thêm.
