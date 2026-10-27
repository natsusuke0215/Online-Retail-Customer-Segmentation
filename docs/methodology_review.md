# Kiểm tra phương pháp: Online Retail → RFM → clustering

## Kết luận hiện tại

Pipeline đã tạo được một **baseline tái lập được** cho bốn model: 391.074 dòng mua hàng hợp lệ, 18.402 hóa đơn, 4.334 khách hàng; tổng Monetary **£8.736.027,64**. Đã có kiểm tra schema, missing, trùng, hủy/hoàn trả, phân phối RFM, ngoại lệ, quan hệ giữa biến, các bảng EDA và test code. Điều này chứng minh dữ liệu đầu vào nhất quán về mặt kỹ thuật, **chưa chứng minh đây là cách phân khúc tối ưu hoặc đã được kiểm định ngoài thời gian**.

## Cơ sở từ nguồn gốc và tài liệu

| Quyết định | Cơ sở | Giới hạn |
| --- | --- | --- |
| Dùng Excel một năm làm baseline, không nối thêm CSV hai năm | [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail) mô tả tập một năm; [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii) mô tả tập hai năm. Đối chiếu nội bộ cho thấy hai nguồn chồng lặp ở năm cuối. | Chọn cửa sổ một năm là quyết định thiết kế, không phải quy định bắt buộc của UCI. |
| `InvoiceNo` bắt đầu bằng `C` là hủy; `CustomerID` là ID, `UnitPrice` tính bằng bảng Anh | Mô tả biến chính thức của [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail). | UCI không hướng dẫn phải bỏ hủy khỏi mọi định nghĩa Monetary. |
| Tạo Recency, Frequency, Monetary ở mức khách hàng | [Nghiên cứu của Chen, Sain và Guo trên chính nhà bán lẻ này](https://researchportal.lsbu.ac.uk/en/publications/data-mining-for-the-online-retail-industry-a-case-study-of-rfm-mo-3/) dùng RFM để phân khúc khách hàng. | Công thức snapshot, chính sách hoàn trả, cửa sổ quan sát và phép chuẩn hóa của project phải được kiểm tra riêng; nghiên cứu không xác nhận từng quyết định code hiện tại. |
| `log1p` rồi `StandardScaler` cho ba feature số | [StandardScaler của scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) đưa mỗi cột về trung bình 0, phương sai 1; log1p làm giảm ảnh hưởng của đuôi dài quan sát được trong EDA. | Log1p là lựa chọn phân tích, không được UCI quy định. StandardScaler **không biến phân phối thành chuẩn** và [nhạy với outlier](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html). Cần so sánh với RobustScaler hoặc lựa chọn khác. |
| Dùng Silhouette, Davies–Bouldin, Calinski–Harabasz khi không có nhãn thật | [Hướng dẫn clustering của scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) mô tả ba chỉ số nội tại và giới hạn của chúng. | Chỉ số ưu ái hình dạng cụm nhất định; không một chỉ số nào chứng minh ý nghĩa kinh doanh. Với DBSCAN, score hiện tính trên điểm không nhiễu nên phải đọc cùng tỷ lệ nhiễu, không xếp hạng trực tiếp chỉ theo score. |
| Không chia ngẫu nhiên giao dịch cho baseline mô tả | Mỗi khách hàng cần một lịch sử giao dịch nguyên vẹn trước snapshot. | Nếu đánh giá khả năng áp dụng vào tương lai, phải dựng snapshot theo thời gian và fit biến đổi trên giai đoạn sớm. [scikit-learn cảnh báo leakage khi fit scaler trên dữ liệu test](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). |

## Kiểm tra trên dữ liệu thực và sửa lỗi vừa thực hiện

- Excel có 541.909 dòng; 135.080 dòng thiếu `CustomerID` (24,9%). Không thể suy ra Customer ID bằng imputation. 98,9% dòng thiếu ID thuộc United Kingdom; điều này cho thấy thiếu dữ liệu có liên hệ với biến quan sát được, nhưng **không đủ để kết luận MCAR/MAR/MNAR**.
- Có 5.268 dòng trùng hoàn toàn. Baseline loại chúng; nếu giữ lại, tổng Monetary tăng **£23.839,01** (khoảng 0,27%). Nguồn không có ID dòng hàng nên chưa thể chứng minh mọi bản ghi giống nhau là lỗi.
- Audit mã hàng phát hiện `23444` có Description “Next Day Carriage”: **79 dòng, £1.200** từng bị tính vào Monetary. Đã thêm mã này vào nhóm phí/điều chỉnh và tạo lại toàn bộ output.
- Nếu giữ các mã phí/điều chỉnh (`POST`, `C2`, `DOT`, `BANK CHARGES`, `M`, `23444`), tổng Monetary tăng **£151.181,25** và thêm 4 khách hàng. Đây là chính sách về định nghĩa “chi tiêu hàng hóa”, không phải dữ kiện được UCI chỉ định.
- Có 8.872 dòng số lượng âm không trùng, có ID; tổng `Amount` là **−£608.689,47**, ảnh hưởng tới 1.589 khách hàng. Trừ đơn giản chúng khỏi gross Monetary sẽ tạo 57 khách hàng có Monetary không dương; một số hoàn trả có thể thuộc đơn mua trước cửa sổ quan sát. Vì vậy baseline hiện là **gross merchandise spend**. Cần ghép hoàn trả với giao dịch gốc hoặc thỏa thuận định nghĩa tiền ròng trước khi khẳng định giá trị khách hàng.
- Feature đã xuất không có NA, mỗi `CustomerID` xuất hiện đúng một lần và tổng Monetary bằng tổng `Amount` của các dòng `included_in_rfm`.

## Việc còn cần làm trước khi gọi kết quả là đã kiểm định

1. Bốn thành viên xuất nhãn cho cùng 4.334 khách hàng; kiểm tra số cụm, tỷ lệ nhiễu, kích thước cụm, ba chỉ số nội tại và hồ sơ RFM/đặc điểm nghiệp vụ của từng cụm. Không chọn model chỉ bằng một score.
2. Chạy độ nhạy với ít nhất: giữ/bỏ dòng trùng, gross/net returns (sau khi xác định chính sách hoàn trả), và `log1p + StandardScaler` so với biến đổi ít nhạy outlier hơn. So sánh số khách đổi cụm và hồ sơ cụm, không chỉ so sánh score.
3. Lập snapshot tại một ngày sớm hơn rồi kiểm tra độ ổn định theo thời gian. Không dùng giao dịch sau ngày snapshot để tính RFM hoặc fit scaler cho snapshot đó.
4. Thống nhất mục tiêu sử dụng cụm với nhóm: phân khúc theo **giá trị mua hàng gộp** hay **giá trị ròng sau hoàn trả**. Hai định nghĩa trả lời hai câu hỏi khác nhau.

`data_cleaned/rfm_model_features.csv` đủ để **bắt đầu thử bốn thuật toán**. Chưa nên dùng cụm cho kết luận kinh doanh hoặc khẳng định model nào tốt nhất khi bốn bước trên chưa hoàn thành.
