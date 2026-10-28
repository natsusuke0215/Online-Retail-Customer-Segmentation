# Bàn giao dữ liệu Online Retail cho kiểm định 90 ngày

## Trạng thái và kế hoạch

| Việc | Trạng thái | Đầu ra |
| --- | --- | --- |
| Làm sạch và kiểm toán giao dịch | Xong | `data_cleaned/transactions_clean.csv`, `quality_report.json` |
| EDA và RFM baseline | Xong | `reports/eda_summary.md`, `reports/figures/`, `rfm_customers.csv` |
| Hai snapshot quá khứ và feature chung | Xong | `data_cleaned/temporal/{development,final}/` |
| Kết quả tương lai tách riêng | Xong | `data_cleaned/temporal/outcomes/` |
| Baseline quyết định 15% trên development | Xong | `reports/temporal/development/simple_rule_baselines.csv` |
| Bốn model, nhãn cụm và chọn tham số | Chờ bốn teammate | `models/outputs/{development,final}/` |
| So sánh cụm với baseline, kiểm tra độ nhạy và demo cuối | Chờ nhãn cụm | `scripts/evaluate_temporal.py`, báo cáo, Streamlit |

## Thiết kế thời gian

| Phase | Giao dịch được dùng tạo RFM | Kết quả mua lại |
| --- | --- | --- |
| Development | Trước `2011-06-01` | `[2011-06-01, 2011-08-30)` (90 ngày) |
| Final | Trước `2011-09-01` | `[2011-09-01, 2011-11-30)` (90 ngày) |

Hai cửa sổ kết quả không chồng nhau. Dữ liệu final chỉ dùng một lần **sau khi** chọn cách làm sạch, feature, scaler, tham số và cách diễn giải từ development. Lượng lịch sử trước hai snapshot khác nhau, nên không coi hai phase là hai lần lặp y hệt để chứng minh độ ổn định cụm.

Chạy `python scripts/run_pipeline.py` trước; sau đó chạy `python scripts/run_temporal_pipeline.py`. Script thứ hai kiểm tra checksum dữ liệu gốc và số dòng của bảng audit trước khi xuất snapshot. Mỗi scaler được fit **chỉ trên khách có giao dịch trước snapshot tương ứng**. Model không được fit lại scaler trên dữ liệu tương lai hoặc ghép outcome vào feature.

## Hợp đồng dữ liệu cho teammate

Mỗi model đọc `data_cleaned/temporal/development/rfm_model_features.csv` khi chọn tham số. File có đúng `CustomerID,Recency_scaled,Frequency_scaled,Monetary_scaled`; chỉ ba cột sau là đầu vào model. `rfm_customers.csv` cùng thư mục giữ RFM gốc để mô tả cụm. `manifest.json` ghi mốc thời gian, số dòng, cột, checksum và quy tắc biến đổi.

Sau khi chốt tham số trên development, chạy cùng phương pháp trên `data_cleaned/temporal/final/rfm_model_features.csv`; không xem outcome final để sửa tham số. Ghi seed, tham số, phiên bản thư viện và thời gian chạy. Xuất hai file `models/outputs/development/<model>.csv` và `models/outputs/final/<model>.csv`, mỗi file đúng một dòng cho mỗi `CustomerID` của phase tương ứng, ít nhất hai cột `CustomerID,cluster`. Tên model: `kmeans`, `agglomerative`, `dbscan`, `gmm`; DBSCAN dùng `-1` cho noise. Không dùng nhãn cụm của final để chọn số cụm mới. Agglomerative/DBSCAN nếu phải lấy mẫu thì báo cáo rõ tập khách được chấm; evaluator hiện yêu cầu nhãn đủ mọi khách và cần quy tắc gán ngoài mẫu nếu muốn so sánh đầy đủ.

**Không chuyển** `data_cleaned/temporal/outcomes/` cho code huấn luyện. Các cột `repurchased_in_90d`, `future_invoices`, `future_revenue` chỉ dùng trong bước đánh giá. Khách mới trong tương lai không có feature quá khứ, nên không bị ép vào cụm cũ.

## Kết quả chuẩn bị dữ liệu đã chạy

| Phase | Dòng quan sát hợp lệ | Khách có feature | Khách cũ mua lại trong 90 ngày | Khách mới trong tương lai |
| --- | ---: | ---: | ---: | ---: |
| Development | 143.150 | 2.716 | 1.337 | 589 |
| Final | 223.099 | 3.314 | 1.859 | 974 |

Các con số là kiểm tra tính khả thi và chất lượng dữ liệu, **chưa phải kết quả model**. EDA của từng snapshot gồm median/phân vị RFM, tương quan Spearman và histogram `log1p(RFM)` tại `reports/temporal/<phase>/`. Audit baseline cho biết 135.080 dòng thiếu CustomerID, 5.268 dòng trùng hoàn toàn và ảnh hưởng của việc giữ dòng trùng/mã phí; các quyết định này vẫn cần phân tích độ nhạy sau khi có cụm.

Baseline development chọn cùng **407/2.716 khách (~15%)** bằng đặc trưng quá khứ. Top Monetary bao phủ **66,41%** doanh thu 90 ngày của khách cũ; điểm RFM ngũ phân vị bao phủ **64,11%**. Đây là phép đo quan sát để đặt chuẩn so sánh, không chứng minh chiến dịch chăm sóc làm tăng doanh thu. `RFM_score` cộng ba điểm 1–5: recency thấp tốt hơn, frequency và monetary cao tốt hơn; hòa điểm được phá bằng Monetary, Frequency, Recency và CustomerID. Tập final chưa chạy baseline quyết định để tránh dùng kết quả cuối khi chọn phương pháp.

## Sau khi nhận nhãn cụm

1. Chạy `python scripts/evaluate_temporal.py --phase development --with-models` để tạo chỉ số nội tại, hồ sơ RFM cụm, tỷ lệ mua lại/doanh thu theo cụm và `cluster_policy_comparison.csv` ở cùng năng lực 15%. Quy tắc cụm ưu tiên cụm có Monetary **quá khứ** trung vị cao, sau đó Frequency cao/Recency thấp; noise DBSCAN được xếp cuối. Nếu cần chọn một phần của cụm cuối, xếp khách trong đó theo RFM quá khứ. Đọc kết quả cùng quy mô cụm và tỷ lệ noise.
2. Chốt model/siêu tham số và quy tắc chọn 15% khách trên development, so với top Monetary và RFM score **ở cùng số khách**. Không xếp cụm theo doanh thu tương lai của final.
3. Chạy `python scripts/evaluate_temporal.py --phase final --with-models` một lần, rồi kiểm tra bootstrap CI cho chênh lệch chính sách và độ nhạy với quy tắc dòng trùng, mã phí, scaler. Không gọi doanh thu quan sát là uplift hay ROI vì dữ liệu không có can thiệp.
4. Sau khi có kết quả đã kiểm tra, tạo demo Streamlit cho cụm, hành vi 90 ngày sau và giới hạn phương pháp.
