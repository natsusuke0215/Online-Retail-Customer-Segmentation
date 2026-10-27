# Kế hoạch khảo sát dữ liệu và chuẩn bị RFM cho phân cụm khách hàng

## 1. Phạm vi và nguồn dữ liệu

Mục tiêu của nhóm là so sánh K-Means, Agglomerative, DBSCAN và GMM trên **cùng một ma trận đặc trưng khách hàng**. Đơn vị phân tích là `CustomerID`, không phải dòng sản phẩm hay hóa đơn.

| Nguồn | Số dòng | Thời gian | Ghi chú |
| --- | ---: | --- | --- |
| `raw_data/online_retail_II.csv` | 1.067.371 | 2009-12-01 đến 2011-12-09 | Hai năm dữ liệu, tên cột `Invoice`, `Price`, `Customer ID` |
| `raw_data/Online Retail.xlsx` | 541.909 | 2010-12-01 đến 2011-12-09 | Một năm dữ liệu, tên cột `InvoiceNo`, `UnitPrice`, `CustomerID` |

Hai file **không được nối trực tiếp**. Sau khi chuẩn hóa tên và kiểu cột, toàn bộ dòng trong Excel đều có trong CSV ở cùng giai đoạn. CSV có thêm 22.524 lần xuất hiện của dòng, trong đó chỉ một dòng có nội dung khác với các dòng Excel. Chọn **Excel làm nguồn chuẩn cho baseline 12 tháng** để kết quả dễ đối chiếu với Online Retail Dataset và tránh cộng trùng. Giữ CSV để phân tích độ ổn định khi dùng lịch sử hai năm; muốn dùng CSV làm nguồn chính thì cần kiểm tra thêm sự khác biệt và trùng lặp.

## 2. Kết quả kiểm tra ban đầu trên Excel

- 541.909 dòng, 8 cột; 4.372 `CustomerID` không rỗng, 25.900 mã hóa đơn, 38 quốc gia. `United Kingdom` chiếm 495.478 dòng (91,4%).
- 135.080 dòng thiếu `CustomerID` (24,9%); 1.454 dòng thiếu `Description` (0,27%). Không có giá trị thiếu ở các cột còn lại.
- 5.268 dòng trùng hoàn toàn; 9.288 dòng có mã hóa đơn bắt đầu bằng `C`; 10.624 dòng có `Quantity < 0`; 2.517 dòng có `UnitPrice <= 0` (trong đó 2 dòng giá âm). Các nhóm này có thể giao nhau, nên không cộng các số liệu để tính tổng dòng loại.
- Sau khi bỏ trùng hoàn toàn, giữ dòng có `CustomerID`, hóa đơn không bắt đầu bằng `C`, `Quantity > 0` và `UnitPrice > 0`: **392.692 dòng, 18.532 hóa đơn, 4.338 khách hàng**. Đây là baseline sơ bộ, trước khi quyết định xử lý các mã phí và điều chỉnh như `POST`, `DOT`, `BANK CHARGES`, `M`.
- Trên baseline sơ bộ, RFM có median: `Recency = 51 ngày`, `Frequency = 2 hóa đơn`, `Monetary = 668,57`. Khoảng 34,4% khách hàng chỉ có một hóa đơn. Giá trị lớn nhất của Frequency là 209 và Monetary là 280.206,02: cần khảo sát đuôi phân phối, không tự động xóa khách hàng giá trị cao.

## 3. Trình tự thực hiện

### A. Khảo sát và kiểm tra chất lượng

1. Đọc Excel với kiểu chuẩn: mã hóa đơn, mã hàng và quốc gia là chuỗi; `CustomerID` là ID, không phải biến số liên tục; ngày là `datetime`; số lượng và giá là số. Chuẩn hóa khoảng trắng ở các cột text nhưng giữ bản gốc để truy vết.
2. Lập báo cáo số dòng, số khách hàng/hóa đơn, khoảng ngày, missing theo cột và theo tháng/quốc gia/loại hóa đơn, trùng hoàn toàn, số lượng/giá không hợp lệ, mã hàng đặc biệt và các mâu thuẫn trong cùng hóa đơn.
3. Xem mẫu các dòng trùng trước khi bỏ: không có ID dòng hàng nên hai dòng giống hệt có thể là hai lần mua thật. Baseline bỏ trùng hoàn toàn; kiểm tra độ nhạy của RFM khi giữ chúng.
4. Không điền `CustomerID` bị thiếu bằng mean, mode hay KNN: không biết khách nào đã mua. Giữ nhóm này trong báo cáo chất lượng, loại khỏi bảng RFM cá nhân. Không thể kết luận MCAR/MAR/MNAR chỉ từ tỷ lệ thiếu; kiểm tra mối liên hệ quan sát được nhưng ghi rõ giới hạn.
5. Phân biệt hóa đơn hủy/hoàn trả, phí vận chuyển, điều chỉnh thủ công và hàng hóa. Baseline RFM tính trên mua hàng dương; lưu thống kê hoàn trả riêng để thử nghiệm chính sách `Monetary` ròng nếu cần.

### B. EDA trước khi biến đổi

1. Theo thời gian: số dòng, hóa đơn, khách hàng hoạt động và doanh thu theo tháng; xem mùa vụ và các tháng đầu/cuối không trọn vẹn.
2. Theo giao dịch: phân phối `Quantity`, `UnitPrice`, giá trị dòng hàng/hóa đơn, tỷ lệ hủy, tỷ lệ thiếu ID và mã hàng đặc biệt. Kiểm tra giá trị âm, bằng 0 và cực lớn bằng bảng mẫu cụ thể.
3. Theo khách hàng: phân phối và boxplot (trục log khi cần) của Recency, Frequency, Monetary; tỷ lệ khách một đơn; tỷ trọng doanh thu của nhóm khách lớn. Dùng IQR/quantile để **gắn cờ** ngoại lệ, không cắt bỏ hàng loạt.
4. Phân tích quan hệ giữa các biến RFM bằng scatterplot/correlation trên thang gốc và thang log. Không có target hay mất cân bằng lớp vì đây là bài toán phân cụm không giám sát. Quốc gia dùng để mô tả cụm và kiểm tra thiên lệch, chưa đưa vào baseline.

### C. Làm sạch, tạo RFM và biến đổi

1. Đóng băng kỳ quan sát baseline từ 2010-12-01 đến 2011-12-09; lấy ngày tham chiếu `2011-12-10`.
2. Tạo `Amount = Quantity × UnitPrice` trên các dòng mua hợp lệ. `Recency`: số ngày từ lần mua cuối đến ngày tham chiếu. `Frequency`: số **hóa đơn duy nhất** mỗi khách. `Monetary`: tổng `Amount` của khách. Kiểm tra hóa đơn chứa nhiều dòng và các dòng phí/điều chỉnh trước khi chốt quy tắc tính tiền.
3. Tạo bảng RFM gốc và một bảng đầu vào model gồm `log1p(Recency)`, `log1p(Frequency)`, `log1p(Monetary)`, sau đó `StandardScaler` dùng chung cho cả bốn model. `StandardScaler` chỉ chuẩn hóa thang đo, không biến phân phối thành chuẩn. So sánh `RobustScaler` hoặc xử lý cực trị theo percentile như phân tích độ nhạy, không thay đổi baseline giữa các model.
4. Không dùng Label Encoding/Target Encoding cho `Country` trong baseline: không có target và mã số quốc gia tạo thứ tự giả. Nếu thử thêm biến phân loại, dùng One-Hot Encoding và đánh giá tác động lên khoảng cách.
5. Mọi phép biến đổi phải tái lập được bằng code và lưu cấu hình, thứ tự cột, mốc thời gian, tiêu chí loại dòng. `CustomerID` chỉ là khóa nối và không đi vào ma trận khoảng cách.

### D. Đánh giá và phối hợp với bốn thành viên làm model

1. Bàn giao một ma trận `X` cố định (mỗi hàng là một khách, ba cột RFM đã biến đổi), cùng `CustomerID` và RFM gốc để gắn nhãn và mô tả cụm. Chốt seed, phiên bản dữ liệu, khoảng tham số và cách lấy mẫu để so sánh công bằng.
2. Viết một hàm đánh giá chung nhận `X` và nhãn cụm: Silhouette Score (cao hơn tốt), Davies–Bouldin Index (thấp hơn tốt), Calinski–Harabasz Score (cao hơn tốt), số cụm, kích thước từng cụm, tỷ lệ nhiễu, thời gian chạy. Chỉ tính ba chỉ số khi có ít nhất hai cụm hợp lệ và đủ điểm; với DBSCAN tính trên điểm không nhiễu, đồng thời báo cáo tỷ lệ nhiễu để tránh so sánh sai.
3. Elbow/Silhouette cho K-Means, dendrogram/linkage cho Agglomerative, lưới `eps`–`min_samples` cho DBSCAN, BIC/AIC cho GMM là phần tối ưu riêng của từng thành viên. Tổng hợp điểm đánh giá chung và hồ sơ RFM từng cụm trong cùng bảng; biểu đồ 2D có thể dùng PCA để trình bày, nhưng chấm điểm trên `X` đầy đủ.
4. Vì đây là clustering không giám sát trên snapshot khách hàng, **không mặc định chia ngẫu nhiên 70/15/15 hay K-Fold**. Nếu cần kiểm tra ổn định ngoài mẫu, dựng thêm snapshot theo mốc thời gian trước đó và so sánh phân bố/hồ sơ cụm. Không dùng giao dịch sau ngày snapshot để tính feature của snapshot đó; không fit scaler trên dữ liệu tương lai. Agglomerative và DBSCAN không có `predict` chuẩn như K-Means/GMM, nên kiểm tra độ ổn định cần quy trình tái phân cụm phù hợp.

## 4. Đầu ra dự kiến

Giữ nguyên `raw_data/`. Pipeline ghi đầu ra vào `data_cleaned/`:

| Tệp | Nội dung |
| --- | --- |
| `transactions_clean.csv` | Dòng giao dịch đã chuẩn hóa, có cờ lý do loại/giữ và `Amount` để kiểm toán |
| `rfm_customers.csv` | Một dòng mỗi `CustomerID`, RFM gốc và metadata phục vụ mô tả cụm |
| `rfm_model_features.csv` | `CustomerID` và ba đặc trưng RFM đã biến đổi, dùng chung cho bốn model |
| `quality_report.json` | Số lượng ở từng bước lọc, missing, trùng lặp, ngoại lệ và quy tắc tạo dữ liệu |
| `preprocessor.joblib` hoặc cấu hình tương đương | Biến đổi được fit và thứ tự feature để dùng lại trong demo |

Nếu sau này bổ sung bảng rộng theo sản phẩm/tháng, chỉ tạo dạng long trong `melting_data/` khi có nhu cầu phân tích cụ thể; RFM hiện chỉ có ba feature nên không cần melt.

**Điểm cần chốt sau EDA:** quy tắc xử lý hóa đơn hủy/hoàn trả; danh sách mã phi hàng hóa; có giữ bản ghi trùng hoàn toàn hay không; dùng một năm hay hai năm làm cửa sổ chính. Các quyết định này phải được áp dụng đồng nhất trước khi so sánh model và đưa vào Streamlit.

## 5. Trạng thái triển khai

Pipeline baseline đã chạy trên Excel, snapshot `2011-12-10`. Quy tắc hiện tại loại hóa đơn hủy/hoàn trả, giao dịch không có ID, dòng trùng hoàn toàn, giá/số lượng không dương và các mã `POST`, `C2`, `DOT`, `BANK CHARGES`, `M`, `23444` khỏi RFM. Kết quả: **391.074 dòng, 18.402 hóa đơn, 4.334 khách hàng**. Median R/F/M là **51 ngày / 2 hóa đơn / 661,42**. Dữ liệu được lưu trong `data_cleaned/`; báo cáo và biểu đồ tại `reports/`. Xem [README.md](../README.md) để chạy lại và bàn giao file cho bốn thành viên làm model. [Đánh giá phương pháp](methodology_review.md) phân biệt baseline kỹ thuật với kết quả đã kiểm định.

Kiểm tra độ nhạy: nếu giữ dòng trùng hoàn toàn, tổng Amount tăng 23.839,01; nếu giữ các mã phí/điều chỉnh, tăng 151.181,25 và thêm 4 khách hàng. Đây là những lựa chọn phân tích cần nêu rõ khi diễn giải cụm, chưa có lý do để cắt khách hàng chi tiêu cao chỉ vì cờ IQR.
