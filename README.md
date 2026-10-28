# Online Retail customer segmentation

Nhóm tạo đặc trưng RFM từ giao dịch Online Retail, so sánh K-Means, Agglomerative, DBSCAN và GMM, rồi kiểm tra các cụm bằng hành vi mua lại trong **90 ngày tiếp theo**. Phần chuẩn bị dữ liệu đã hoàn tất; bốn model và kết luận cuối đang chờ nhóm triển khai.

## Bắt đầu

Dùng Python 3.11+; tải `Online Retail.xlsx` theo [hướng dẫn nguồn dữ liệu](raw_data/README.md) và đặt trong `raw_data/`. Từ thư mục gốc:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts/run_pipeline.py
.\.venv\Scripts\python scripts/run_temporal_pipeline.py
.\.venv\Scripts\python scripts/build_eda_report.py
.\.venv\Scripts\python scripts/evaluate_temporal.py --phase development
.\.venv\Scripts\python -m pytest -q
```

`run_pipeline.py` tạo bảng giao dịch đã kiểm toán; `run_temporal_pipeline.py` tạo hai snapshot cho model; `build_eda_report.py` tái tạo [báo cáo EDA và 6 biểu đồ](docs/eda/README.md). Dữ liệu gốc và các output lớn trong `data_cleaned/`, `reports/` không theo Git; **bản EDA chọn lọc trong `docs/eda/` có trên GitHub**.

## Trạng thái sáu bước

| Bước | Trạng thái |
| --- | --- |
| Thu thập và làm sạch | Xong, có audit và báo cáo chất lượng |
| EDA | Xong cho nguồn và tập development; [xem insight và plot](docs/eda/README.md) |
| Feature engineering | Xong: RFM, `log1p`, scaler fit riêng theo snapshot |
| Chọn feature và chia dữ liệu | Baseline ba feature đã chốt; development/final chia theo thời gian. Kiểm tra độ nhạy feature/scaler còn chờ model |
| Xây bốn model | Teammate đang triển khai |
| Đánh giá cuối và demo | Evaluator đã có; kết luận và demo chờ nhãn cụm |

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
docs/                           EDA có hình, hướng dẫn dữ liệu và phương pháp
tests/                          Kiểm tra logic quan trọng
```

**Đọc trước khi làm model:** [Hướng dẫn dữ liệu cho teammate](docs/team_data_guide.md). Tài liệu [bàn giao chi tiết](docs/temporal_handoff.md) ghi mốc thời gian và các bước đánh giá sau khi có nhãn cụm. File `data_cleaned/rfm_model_features.csv` ở thư mục gốc là snapshot **cuối năm để mô tả**, không dùng cho kiểm định 90 ngày ở mốc 01/09.
