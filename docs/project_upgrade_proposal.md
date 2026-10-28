# Đề xuất nâng cấp đề tài cho môn học và portfolio Data Scientist

> Phần dữ liệu của đề xuất đã được triển khai thành hai snapshot và baseline development. Xem [bàn giao hiện hành](temporal_handoff.md). Bốn model và đánh giá cuối vẫn chờ nhãn cụm của nhóm.

## Nhận định

Mô tả đề tài trong danh sách lớp đang nói về **tuổi, thu nhập, mức độ chi tiêu**, trong khi dữ liệu Online Retail của nhóm **không có tuổi hay thu nhập**. Trước tiên cần sửa đề cương chính thức cho khớp dữ liệu giao dịch và RFM; đây là lỗi trình bày quan trọng hơn việc thêm một thuật toán mới.

Đề tài hiện tại mạnh hơn bài phân cụm trên vài cột nhân khẩu học: nhóm đã xử lý 541.909 dòng giao dịch thật, missing Customer ID, hóa đơn hủy, dòng trùng, phí vận chuyển và dữ liệu ở mức khách hàng. Điểm còn yếu nếu chỉ dừng ở RFM + bốn thuật toán + Silhouette là **chưa kiểm tra cụm có liên hệ với hành vi kinh doanh trong tương lai hay không**. [UCI](https://archive.ics.uci.edu/dataset/352/online%2Bretail) đã giới thiệu bộ dữ liệu cùng [nghiên cứu RFM-clustering của Chen và cộng sự](https://researchportal.lsbu.ac.uk/en/publications/data-mining-for-the-online-retail-industry-a-case-study-of-rfm-mo-3/), nên việc chỉ lặp lại RFM + K-Means khó tạo điểm khác biệt.

Một [mô tả vị trí Data Scientist của Microsoft](https://careers.microsoft.com/v2/global/en/students?jobSeqNo=1498253) nhấn mạnh chất lượng dữ liệu, đánh giá mô hình theo mục tiêu kinh doanh và truyền đạt tác động. Đây là ví dụ về tín hiệu kỹ năng mà portfolio có thể thể hiện, không phải chuẩn chung của mọi vị trí.

## Phương án khuyến nghị: giữ bốn model, thêm kiểm định theo thời gian

**Tên đề tài gợi ý:** *Phân khúc có giúp ra quyết định tốt hơn quy tắc RFM đơn giản? Kiểm định trên hành vi mua sắm tương lai* / *Do Customer Clusters Improve Decisions Beyond Simple RFM Rules?*.

**Mô tả đề cương có thể gửi giảng viên:** Nhóm sử dụng dữ liệu giao dịch Online Retail để xây dựng đặc trưng RFM ở mức khách hàng, xử lý hóa đơn hủy, dữ liệu thiếu và các giao dịch bất thường. Trên cùng bộ đặc trưng, nhóm so sánh K-Means, Agglomerative Clustering, DBSCAN và Gaussian Mixture Model bằng các chỉ số phân cụm nội tại, tỷ lệ nhiễu và khả năng diễn giải cụm. Để kiểm tra giá trị sử dụng, nhóm chỉ dùng giao dịch trước một mốc thời gian để tạo cụm, sau đó đo tỷ lệ mua lại và giá trị mua hàng trong 90 ngày tiếp theo. Tại cùng năng lực chăm sóc 15% khách hàng, nhóm so sánh lựa chọn dựa trên cụm với quy tắc RFM/Monetary đơn giản, đồng thời kiểm tra độ nhạy của kết quả. Nghiên cứu phân biệt ích lợi của cụm cho diễn giải danh mục với sức dự đoán ở mức cá nhân; không suy diễn tác động nhân quả của chiến dịch tiếp thị từ dữ liệu quan sát.

**Câu hỏi chính:** Các cụm được tạo từ giao dịch quá khứ có phân biệt được tỷ lệ mua lại và giá trị mua hàng trong 90 ngày sau không? Cụm nào có hồ sơ đủ rõ để mô tả một nhóm khách hàng, và kết quả có bền khi đổi cách làm sạch/chuẩn hóa?

1. **Quan sát**: dùng giao dịch từ 2010-12-01 đến hết 2011-08-31 để tính RFM theo snapshot `2011-09-01`. Chỉ dùng khoảng này để fit scaler, chọn tham số và phân cụm.
2. **Giữ riêng tương lai**: dùng khoảng nửa kín `[2011-09-01, 2011-11-30)` (đến hết 29/11, đúng 90 ngày) để đo `repurchased_in_90d`, số hóa đơn và doanh thu tương lai cho khách đã xuất hiện trong giai đoạn quan sát. Không đưa ba biến này vào đầu vào clustering. Không dùng tháng 12/2011 vì dữ liệu dừng ngày 09/12.
3. **So sánh có baseline**: bốn thuật toán hiện tại so với quy tắc RFM theo quantile. Báo cáo Silhouette, Davies–Bouldin, Calinski–Harabasz, tỷ lệ nhiễu và kích thước cụm; sau đó báo cáo tỷ lệ mua lại/doanh thu tương lai mỗi cụm, kèm quy mô cụm và khoảng tin cậy khi phù hợp. Chỉ số nội tại không đủ để quyết định ý nghĩa kinh doanh; các chỉ số này cũng có [thiên hướng theo hình dạng cụm](https://scikit-learn.org/stable/modules/clustering.html).
4. **Kiểm tra độ nhạy**: giữ/bỏ dòng trùng, quy tắc hoàn trả, StandardScaler/RobustScaler; so sánh hồ sơ cụm và mức thay đổi nhãn trên cùng khách hàng. Không tự động cắt bỏ khách chi tiêu cao.
5. **Tránh leakage**: dùng giai đoạn tương lai một lần để báo cáo cuối. Nếu cần phát triển tham số dựa trên kết quả ngoài thời gian, tạo một mốc sớm hơn cho validation, ví dụ quan sát đến hết 2011-05 rồi đo mua lại trong `[2011-06-01, 2011-08-30)` (90 ngày). [scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) khuyến cáo fit các phép biến đổi chỉ trên dữ liệu huấn luyện.

**Khả thi trên dữ liệu hiện có:** mốc `2011-09-01` có **3.314 khách hàng** và **223.099 dòng mua hàng hợp lệ**. Trong 90 ngày tiếp theo, **1.859/3.314 khách (56,1%)** mua lại; có **974 khách mới** chỉ xuất hiện ở giai đoạn sau, cần báo cáo riêng thay vì ép vào cụm cũ. Đây là thống kê sơ bộ để thiết kế thí nghiệm, chưa phải kết quả model.

## Bổ sung sau kết quả thăm dò của nhóm: benchmark quyết định

Kết quả K-Means k=4, η² và AUC được cung cấp trong trao đổi là **kết quả do nhóm báo**, chưa có file nhãn/code trong repo để tái lập. Có hai điểm cần sửa trước khi dùng chúng trong báo cáo:

- Nhóm báo 3.309 khách tại mốc 31/08; pipeline hiện tại có **3.314**. Cần đối chiếu danh sách `CustomerID`, cutoff ngày, dòng trùng và các mã phí trước khi so bất cứ con số nào.
- Hai mốc 01/09 và 09/09 cách nhau 8 ngày. Nếu cùng dùng cửa sổ tương lai 91 ngày, hai tập kiểm tra **trùng nhau 83/91 ngày (~91%)**. Kết quả giống nhau chưa chứng minh độ ổn định ngoài thời gian. Nên dùng hai cửa sổ kết quả không chồng lặp, ví dụ snapshot 01/06 và 01/09, mỗi lần 180 ngày quan sát và 90 ngày tương lai. So sánh cụm trên khách chung bằng [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) sau khi ghép đúng tập khách, nhưng nhớ rằng hành vi khách cũng có thể thay đổi thật.

Một đối chứng đơn giản đã chạy trên **pipeline hiện tại**: tại snapshot 01/09, chọn 497 khách (15%) có Monetary quá khứ cao nhất thì nhóm này tạo **62,7% doanh thu 91 ngày sau**. Con số này cao hơn **60,5%** mà nhóm báo cho cụm VIP 498 khách, dù hai kết quả **chưa hoàn toàn cùng bộ lọc**. Vì vậy câu “VIP 15% tạo 60% doanh thu tương lai” chứng minh doanh thu tập trung, **chưa chứng minh clustering tạo thêm giá trị so với một luật rất đơn giản**.

### Câu hỏi kinh doanh nên chốt

> Nếu bộ phận chăm sóc chỉ có năng lực theo dõi 15% khách hàng, dùng cụm RFM có chọn được danh sách khách có giá trị mua hàng tương lai tốt hơn quy tắc xếp hạng bằng Monetary/RFM đơn giản không? Và nếu không, cụm còn hữu ích cho việc nào?

So sánh tại **cùng số khách được chọn**: (1) ngẫu nhiên, (2) top Monetary hoặc RFM score, (3) cụm VIP của K-Means/GMM, (4) điểm dự đoán RFM dạng số nếu nhóm làm supervised baseline. Chỉ số chính là tỷ lệ doanh thu tương lai nằm trong danh sách được chọn; bổ sung tỷ lệ mua lại, độ ổn định qua mốc thời gian, phân phối chi tiêu và khoảng tin cậy bootstrap cho **chênh lệch giữa các chính sách**. Chọn tham số trên giai đoạn phát triển, giữ mốc cuối làm kiểm tra cuối. Nếu quy tắc đơn giản bằng hoặc hơn cluster, khuyến nghị dùng quy tắc cho xếp hạng cá nhân và dùng cụm cho truyền đạt/hồ sơ danh mục.

Đây là bài toán **ưu tiên chăm sóc khách có giá trị**, không đo được “giữ chân với chi phí thấp nhất”. Dữ liệu không có biên lợi nhuận, chi phí liên hệ hay nhóm được/không được can thiệp. Muốn ước lượng tác động tăng retention hoặc ROI, cần thử nghiệm ngẫu nhiên sau này; dashboard hiện chỉ nên hiển thị mô phỏng theo giả định, không trình bày doanh thu quan sát như uplift.

Khi nhãn cụm được tạo hoàn toàn từ RFM, nó không mang thông tin gốc mới ngoài RFM. Vì thế AUC của RFM + cluster không tăng là kết quả hợp lý; phần supervised, nếu làm, nên được trình bày như **đối chứng để kiểm tra giá trị bổ sung**, không phải lời hứa cải thiện dự đoán. K-Means có thể là model chính vì đơn giản và dễ diễn giải; dùng GMM làm đối chứng xác suất, Agglomerative/DBSCAN để kiểm tra độ nhạy hình dạng cụm. Chỉ nâng GMM thành model chính nếu có lợi ích đo được.

## Phân công không làm hỏng kế hoạch nhóm

| Người/phần | Thay đổi cần thiết |
| --- | --- |
| K-Means, Agglomerative, DBSCAN, GMM | Vẫn triển khai thuật toán và tối ưu tham số như đã phân công. Mỗi người dùng cùng snapshot đầu vào, xuất `CustomerID,cluster` và mô tả cụm. |
| Tiền xử lý, đánh giá, tích hợp | Tạo snapshot quá khứ và bảng kết quả tương lai tách biệt; viết evaluator đo kết quả nội tại, tỷ lệ mua lại/doanh thu tương lai, so sánh quyết định tại cùng năng lực 15% và độ nhạy; tích hợp vào báo cáo và Streamlit. |
| Cả nhóm | Chọn cụm có ý nghĩa dựa trên quy mô, hồ sơ, sự ổn định và mối liên hệ với kết quả tương lai; không chọn model chỉ vì Silhouette cao nhất. |

## Phần mở rộng chỉ làm nếu còn thời gian

- Thử 1–2 đặc trưng có giải thích rõ, như thời gian từ lần mua đầu hoặc giá trị đơn hàng trung bình, rồi làm ablation so với RFM gốc. Không thêm nhiều cột chỉ để tăng vẻ phức tạp.
- Một baseline supervised nhỏ dự đoán mua lại 90 ngày có thể **kiểm tra** liệu nhãn cụm có cải thiện dự đoán so với RFM số hay không; không nên kỳ vọng nó bổ sung thông tin mới vì cụm được tạo từ chính RFM. Cần split theo thời gian, đánh giá calibration/PR-AUC hoặc ROC-AUC phù hợp và không dùng nhãn tương lai khi tạo cụm. Đây là đối chứng portfolio, không cần để hoàn thành bài clustering.
- Dashboard thể hiện hành trình **dữ liệu → cụm → hành vi 90 ngày sau → giới hạn phương pháp**. Không tuyên bố chiến dịch marketing tạo uplift vì bộ dữ liệu không có can thiệp hay đối chứng.

## Cách viết CV khi có kết quả thật

> Built a reproducible RFM pipeline from 541k retail transactions; benchmarked four clustering methods against a simple customer-prioritization rule at equal capacity, using out-of-time repurchase and revenue coverage with leakage checks and sensitivity analysis.

Sau khi có kết quả, thay bằng số đo thực tế như số khách hàng, số cụm, mức chênh lệch tỷ lệ mua lại và độ ổn định. Không ghi “tăng doanh thu/retention” nếu không có thử nghiệm can thiệp chứng minh tác động.
