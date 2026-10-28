# Online Retail customer segmentation

Nhóm tạo đặc trưng RFM từ giao dịch Online Retail, so sánh K-Means, Agglomerative, DBSCAN và GMM, rồi kiểm tra các cụm bằng hành vi mua lại trong **90 ngày tiếp theo**. Phần chuẩn bị dữ liệu đã hoàn tất; bốn model và kết luận cuối đang chờ nhóm triển khai.

## Bắt đầu

Dùng Python 3.11+; tải `Online Retail.xlsx` theo [hướng dẫn nguồn dữ liệu](raw_data/README.md) và đặt trong `raw_data/`. Từ thư mục gốc:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts/run_pipeline.py
.\.venv\Scripts\python scripts/run_temporal_pipeline.py
.\.venv\Scripts\python scripts/evaluate_temporal.py --phase development
.\.venv\Scripts\python -m pytest -q
```

`run_pipeline.py` tạo bảng giao dịch đã kiểm toán và EDA; `run_temporal_pipeline.py` tạo hai snapshot cho model. Dữ liệu gốc, dữ liệu đã xử lý và báo cáo sinh ra được `.gitignore` loại khỏi Git; teammate chạy hai script để tái tạo chúng trên máy mình.

## Cấu trúc

```text
raw_data/                       Online Retail.xlsx (không theo Git)
data_cleaned/                   Audit giao dịch, báo cáo chất lượng
  temporal/development/         RFM và feature để chọn model
  temporal/final/               RFM và feature để đánh giá cuối
  temporal/outcomes/            Kết quả 90 ngày; không dùng để train
src/retail_segmentation/        Cleaning, EDA, feature, evaluation
scripts/                       Lệnh chạy pipeline và đánh giá
models/outputs/{development,final}/   Nhãn cụm do teammate tạo
reports/                        EDA và bảng so sánh (không theo Git)
docs/                           Hướng dẫn dữ liệu và phương pháp
tests/                          Kiểm tra logic quan trọng
```

**Đọc trước khi làm model:** [Hướng dẫn dữ liệu cho teammate](docs/team_data_guide.md). Tài liệu [bàn giao chi tiết](docs/temporal_handoff.md) ghi mốc thời gian và các bước đánh giá sau khi có nhãn cụm. File `data_cleaned/rfm_model_features.csv` ở thư mục gốc là snapshot **cuối năm để mô tả**, không dùng cho kiểm định 90 ngày ở mốc 01/09.
