# Kế hoạch mới: phân khúc khách hàng và đo hiệu quả gia tăng của email

> **Trạng thái:** kế hoạch sau khảo sát ban đầu; chưa tạo dữ liệu đã xử lý hoặc huấn luyện model. Dữ liệu gốc trong `raw_data/` được giữ nguyên.

## 1. Câu hỏi và phạm vi

**Câu hỏi kinh doanh:** Với khách hàng có lịch sử mua sắm khác nhau, nên gửi email giới thiệu hàng Nam, email giới thiệu hàng Nữ, hay không gửi email để **tăng thêm** lượt mua/chi tiêu so với không gửi? Nếu năng lực liên hệ có hạn, phân khúc nào đáng ưu tiên cho chiến dịch tiếp theo?

**Vai trò của clustering:** tạo nhóm bằng thông tin **trước chiến dịch**. **Vai trò của thử nghiệm:** so sánh kết quả giữa các nhóm được gán ngẫu nhiên nhận email Nam, email Nữ và không nhận email *trong từng cụm*. Không gọi nhóm mua nhiều là “được email tác động mạnh” nếu chưa so với nhóm không nhận email.

Tên đề tài có thể giữ trong khung đã đăng ký, nhưng sửa mô tả biến: *Phân khúc khách hàng theo hành vi mua sắm để lựa chọn chiến dịch email hiệu quả*. Dữ liệu này không có tuổi, thu nhập, lịch sử từng hóa đơn, hay ngày giao dịch để tính RFM đầy đủ.

Nguồn gốc và định nghĩa biến: [MineThatData E-Mail Analytics and Data Mining Challenge do Kevin Hillstrom công bố](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html?m=0).

## 2. Tổng quan dữ liệu đã khảo sát

File: `raw_data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`; SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.

| Nhóm cột | Cột | Ý nghĩa và vai trò |
| --- | --- | --- |
| Trước chiến dịch | `recency`, `history`, `history_segment`, `mens`, `womens`, `zip_code`, `newbie`, `channel` | Chỉ các cột này mới được cân nhắc làm feature phân cụm. `recency` là **số tháng** từ lần mua gần nhất. `history_segment` là phiên bản chia khoảng của `history`. |
| Phân nhóm thử nghiệm | `segment` | `Mens E-Mail`, `Womens E-Mail`, `No E-Mail`; **không** đưa vào feature phân cụm. |
| Sau chiến dịch | `visit`, `conversion`, `spend` | Kết quả trong **hai tuần** sau email; **không** đưa vào feature phân cụm. |

File có **64.000 hàng × 12 cột**, không có giá trị NA và không có cột định danh khách hàng. Ba nhánh thử nghiệm lần lượt có **21.307 / 21.387 / 21.306** khách (Nam / Nữ / không email). Có **6.562 hàng giống hoàn toàn**, nhưng không có ID để chứng minh đó là bản ghi nhập trùng: nhiều khách có thể có cùng đặc trưng và kết quả. **Không xóa tự động.** Tạo `row_id` theo vị trí dòng chỉ để nối dữ liệu và truy vết; không đưa `row_id` vào model.

`history` lệch phải (median **158,11**, p99 **1.218,66**, max **3.345,93**); `recency` nằm trong **1–12 tháng**. `conversion` trung bình khoảng **0,9%** và `spend` bằng 0 ở khoảng **99,1%** khách, nên phân tích theo cụm nhỏ có độ bất định lớn. Kiểm tra ban đầu cho thấy không có dòng `conversion=1` mà `visit=0`, cũng không có `spend>0` khi `conversion=0`.

| Nhánh | Visit | Conversion | Spend trung bình **trên mọi khách được gán** |
| --- | ---: | ---: | ---: |
| Không email | 10,62% | 0,57% | $0,65 |
| Email Nam | 18,28% | 1,25% | $1,42 |
| Email Nữ | 15,14% | 0,88% | $1,08 |

Đây là **thống kê toàn bộ file để hiểu nguồn**, chưa có kết quả theo cụm, khoảng tin cậy, chọn model hay chính sách gửi email. Không khẳng định nhóm nào nên nhận email nào từ bảng tổng này.

## 3. Thu thập, chuẩn hóa và kiểm tra chất lượng

1. Giữ nguyên CSV gốc, xác nhận số dòng/cột, kiểu dữ liệu, giá trị hợp lệ của biến nhị phân và ba nhánh `segment`; lập báo cáo nguồn và checksum.
2. Kiểm tra NA, chuỗi rỗng, nhãn lạ, hàng giống nhau, khoảng giá trị và logic `conversion → visit`, `spend>0 → conversion`. Hiện không có NA nên **không impute**; MCAR/MAR/MNAR không phải kết luận cần đưa ra cho file này. Nếu gặp NA ở lần nhập khác, tìm nguyên nhân theo biến/nhánh trước khi quyết định cách xử lý.
3. Chuẩn hóa nhãn `zip_code` đang viết `Surburban` thành `Suburban` trong bảng đã xử lý, lưu giá trị gốc để kiểm toán. Không tự tạo vị trí địa lý từ nhãn Urban/Suburban/Rural.
4. Không xóa khách chi tiêu lịch sử cao chỉ vì IQR/Z-score; gắn cờ và kiểm tra ảnh hưởng. Không coi DBSCAN noise là gian lận, spam hoặc khách nên loại khỏi email.
5. Do không có ID khách và dữ liệu đã ở mức một hàng/khách, **không melt** bảng 12 cột; `melting_data/` chỉ tạo nếu về sau có nhu cầu rõ ràng.

## 4. EDA theo đúng trình tự

**Trước khi chọn cụm:** histogram/boxplot cho `recency`, `history`; tần suất `mens`, `womens`, `newbie`, `channel`, `zip_code`; tương quan và các tổ hợp hành vi. Kiểm tra `history_segment` có mang thông tin mới ngoài `history` hay không (dự kiến không). Xem p99, max và một số hàng ngoại lệ nhưng không quyết định xóa bằng mắt.

**Kiểm tra thử nghiệm:** quy mô ba nhánh, mức cân bằng các đặc trưng trước chiến dịch giữa nhánh (bảng và standardized mean difference), tỷ lệ `visit`/`conversion`, `spend` **trung bình trên mọi khách** gồm cả giá trị 0, và khoảng tin cậy của chênh lệch email Nam/Nữ so với không email. Không tính “chi tiêu trung bình chỉ ở người đã mua” để đo tác động của email, vì việc đã mua là kết quả sau can thiệp.

**Giới hạn xem dữ liệu:** báo cáo chất lượng và số liệu tổng nguồn có thể xem toàn file. EDA chi tiết về tương tác đặc trưng–kết quả, chọn số cụm và chọn chiến dịch dùng **development**; tập test không dùng để chọn giả thuyết hoặc tham số.

## 5. Feature engineering và preprocessing chung

**Baseline đầu vào:** `recency`, `log1p(history)`, `mens`, `womens`, `newbie`, và One-Hot của `channel`. `zip_code` dùng để mô tả cụm và chạy thử độ nhạy; không đưa vào baseline vì không trực tiếp là hành vi mua. `history_segment` loại khỏi baseline vì là cách chia khoảng của `history`, nếu đưa cả hai vào sẽ nhân đôi trọng số của chi tiêu lịch sử.

- Fit biến đổi trên development: chuẩn hóa hai biến số sau log/kiểm tra outlier; giữ biến nhị phân là 0/1; One-Hot cho `channel`, xử lý giá trị mới khi transform. `StandardScaler` đưa cột số về trung bình 0/độ lệch chuẩn 1, **không làm phân phối trở thành chuẩn**. Thử RobustScaler/đổi trọng số nhóm biến như phân tích độ nhạy vì khoảng cách Euclid có thể bị feature chi phối.
- Không Label Encoding cho `channel`/`zip_code` vì tạo thứ tự giả; không Target Encoding theo `conversion` cho clustering vì sẽ biến đầu vào không giám sát thành có nhãn và tăng nguy cơ leakage.
- Không tạo `dayofweek`, `month` hay khoảng cách lat/long: file không có timestamp hoặc tọa độ. Không gọi bộ feature này là RFM đầy đủ vì thiếu `Frequency`.
- Lưu một `feature_manifest` và một preprocessor dùng chung; tất cả bốn model lấy cùng thứ tự cột, cùng hàng và cùng split. `segment`, `visit`, `conversion`, `spend`, `row_id` không nằm trong ma trận khoảng cách.

## 6. Feature selection và chia dữ liệu

Đây là dữ liệu chụp tại một thời điểm, **không có trục thời gian từng khách để time-based split**. Tách khách ngẫu nhiên, có stratify theo **nhánh email** để giữ thí nghiệm cân bằng: đề xuất **60% development / 40% test cuối**. Trong development dùng resampling/validation để chọn feature, scaler, tham số và chính sách email; test giữ nguyên cho một lần đánh giá sau khi chốt. Vì conversion hiếm, không ép 70/15/15 làm tập test quá nhỏ cho từng tổ hợp cụm × email.

Feature selection baseline dựa vào ý nghĩa nghiệp vụ, trùng thông tin (`history_segment`–`history`), near-zero variance và độ ổn định cụm. Nếu dùng `conversion` hoặc `spend` để chọn feature, đó là một nhánh supervised riêng và chỉ được thực hiện trong development, không được gọi là clustering thuần không giám sát.

## 7. Bàn giao và đánh giá bốn thuật toán

| Thành viên | Phần mô hình | Điểm cần lưu ý với Hillstrom |
| --- | --- | --- |
| 1 — K-Means | Chọn `k` bằng Elbow/Silhouette, mô tả centroid và quy mô cụm. | Ứng viên chính vì đơn giản, có `predict` để gán khách test. Kiểm tra seed và khả năng diễn giải. |
| 2 — Agglomerative | Ward/complete/average khi phù hợp; dendrogram trên mẫu. | **64.000 hàng quá lớn cho dendrogram/full Agglomerative thông thường**. Dùng một mẫu development cố định, ghi rõ chi phí/tính đại diện. Không coi kết quả trên mẫu là đánh giá full-data. |
| 3 — DBSCAN | Khảo sát `eps`, `min_samples`, tỷ lệ noise và hình dạng cụm. | Nhạy với thang đo và dữ liệu hỗn hợp; noise là điểm ít hàng xóm, không phải khách xấu. Không có `predict` chuẩn cho khách test. |
| 4 — GMM | Chọn số thành phần bằng BIC/AIC, so xác suất thuộc cụm với K-Means. | Ma trận có cả biến nhị phân/One-Hot, nên xem GMM là đối chứng thực nghiệm; kiểm tra thành phần nhỏ và xác suất bất định. Có `predict` cho khách test. |
| 5 — Tiền xử lý/đánh giá/tích hợp | Data quality, EDA, feature matrix/splits, evaluator, bảng so sánh, demo. | Viết phần **so tác động email với control theo từng cụm**, CI và giá trị chính sách trên test; không chỉ ba chỉ số hình học. |

**Bảng đánh giá nội tại** trên cùng tập khách và ma trận feature: Silhouette, Davies–Bouldin, Calinski–Harabasz khi có ít nhất hai cụm hợp lệ; số cụm, kích thước cụm, tỷ lệ noise, độ ổn định qua seed/mẫu. Các chỉ số này không đo tác động email và có thiên hướng theo hình dạng cụm. Agglomerative/DBSCAN không tự gán cụm cho khách test; nếu muốn so tác động ngoài mẫu, cần quy tắc gán trước, được mô tả và kiểm tra riêng. Nếu không, chỉ báo cáo chúng như so sánh khám phá, không trộn vào bảng giá trị chính sách ngoài mẫu.

**Bảng giá trị kinh doanh** trên test cho model có quy tắc gán cụm: với mỗi cụm báo cáo số khách và số lượt mua ở **từng nhánh**, chênh lệch `conversion` và `spend` trung bình trên **mọi khách được gán** giữa email Nam/Nữ và No E-Mail, kèm CI bootstrap hoặc phương pháp tương đương. Không tuyên bố phân khúc có hiệu ứng khác nhau khi khoảng tin cậy quá rộng. Nếu thử nhiều cụm/nhánh, nêu rủi ro phát hiện giả do nhiều phép so sánh.

**Chính sách gửi email:** chọn Nam/Nữ/không gửi cho mỗi cụm **chỉ từ development**; đánh giá chính sách đã chốt trên test nhờ phân nhóm email ngẫu nhiên (ước lượng theo xác suất gán nhánh hoặc phương pháp off-policy tương đương). So với hai baseline: gửi cùng một loại email cho tất cả và quy tắc đơn giản theo `history`/`recency`. Câu “chọn/loại 10.000 khách” là bài toán năng lực liên hệ: xếp theo **mức tăng kỳ vọng do email**, không mặc định theo chi tiêu lịch sử hoặc loại hết DBSCAN noise. Nếu không có chi phí email/biên lợi nhuận, kết luận về **incremental conversion/spend**, không gọi là lợi nhuận hoặc ROI.

## 8. Cấu trúc đầu ra dự kiến khi bắt đầu triển khai

```text
raw_data/                      CSV gốc, không chỉnh sửa
docs/                          Kế hoạch, nguồn tham khảo, quyết định phương pháp
src/                           Cleaning, feature pipeline, evaluation
models/                        File của bốn thành viên
data_cleaned/                  row_id + dữ liệu chuẩn hóa, split, feature matrix, preprocessor, quality report
reports/                       EDA, đối chiếu baseline, kết quả cụm và hiệu quả email
app.py                         Demo Streamlit sau khi có kết quả đã kiểm tra
```

Không tạo các file trên hoặc chia data ngay trong giai đoạn **viết plan**. Khi triển khai, `data_cleaned/` sẽ tách rõ bảng **pre-campaign feature** khỏi bảng **treatment/outcome**, giữ một `row_id` để nối; `raw_data/` không bị ghi đè.

## 9. Điều kiện để kết luận

Project chỉ hoàn thành khi trả lời được: (1) các nhóm có đủ lớn và diễn giải được không; (2) email Nam/Nữ có **tác động tăng thêm** khác nhau theo nhóm hay không, với độ bất định; (3) chính sách theo cụm có tốt hơn gửi chung hoặc quy tắc đơn giản trên test không; (4) kết luận thay đổi thế nào khi đổi scaler/số cụm. Nếu câu trả lời là “phân cụm không cải thiện chính sách”, đó vẫn là kết luận có ích và trung thực.

Giới hạn: dữ liệu từ **2008**, không có khách ID, không có giao dịch chi tiết hay timestamp, không có chi phí email/biên lợi nhuận và chỉ theo dõi **hai tuần**. Không suy rộng trực tiếp kết quả sang khách hàng hoặc chiến dịch hiện nay.
