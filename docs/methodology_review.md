# Quy tắc dữ liệu và giới hạn phương pháp

## Nguồn và đơn vị phân tích

Project dùng [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail), gồm 541.909 dòng giao dịch từ 01/12/2010 đến 09/12/2011. [Chen, Sain và Guo](https://researchportal.lsbu.ac.uk/en/publications/data-mining-for-the-online-retail-industry-a-case-study-of-rfm-mo-3/) dùng RFM để phân khúc trên nguồn này. Đơn vị đưa vào model là **khách hàng**, không phải dòng sản phẩm. Dữ liệu không có tuổi, thu nhập hay thử nghiệm marketing.

## Làm sạch và tạo feature

- Chuẩn hóa mã, ngày và số; giữ toàn bộ dòng cùng cờ lý do loại trong `transactions_clean.csv` để kiểm toán.
- Không điền 135.080 `CustomerID` thiếu vì không thể suy ra danh tính. Loại hóa đơn hủy, số lượng/giá không dương, dòng trùng hoàn toàn và các mã phí/điều chỉnh `POST`, `C2`, `DOT`, `BANK CHARGES`, `M`, `23444` khỏi RFM.
- Baseline cuối năm còn 391.074 dòng mua hàng, 18.402 hóa đơn và 4.334 khách. Đây là **gross merchandise spend**: dòng hoàn trả không bị trừ vào Monetary. Nguồn không có ID dòng hàng đủ để chứng minh mọi dòng giống hệt là lỗi; giữ dòng trùng sẽ tăng tổng Amount £23.839,01. Giữ mã phí/điều chỉnh tăng £151.181,25 và thêm 4 khách. Cần kiểm tra độ nhạy trước khi diễn giải kết quả kinh doanh.
- `Recency` là số ngày từ lần mua cuối đến snapshot, `Frequency` là số hóa đơn duy nhất, `Monetary` là tổng `Quantity × UnitPrice` hợp lệ. Ba cột được `log1p` rồi `StandardScaler`; scaler chỉ fit trên khách có lịch sử **trước** snapshot. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) chuẩn hóa trung bình/phương sai, không làm phân phối trở thành chuẩn.

## Kiểm định theo thời gian

Development dùng lịch sử trước 01/06/2011; final dùng lịch sử trước 01/09/2011. Mỗi phase đo mua lại trong 90 ngày sau. Hai cửa sổ kết quả không chồng nhau. Các outcome nằm riêng khỏi feature và không được dùng để chọn tham số trên final. Xem [hướng dẫn dữ liệu](team_data_guide.md) và [bàn giao chi tiết](temporal_handoff.md).

Silhouette, Davies–Bouldin và Calinski–Harabasz chỉ đo hình học cụm; [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) nêu giới hạn của các thước đo này. Báo cáo thêm kích thước cụm, tỷ lệ noise, hồ sơ RFM, tỷ lệ mua lại và doanh thu tương lai. So lựa chọn 15% khách với quy tắc Monetary/RFM đơn giản ở cùng quy mô. Do không có nhóm can thiệp, chênh lệch doanh thu quan sát **không** phải tác động nhân quả của việc chăm sóc khách.

Chưa có nhãn của bốn model; kết luận model nào tốt hơn, độ nhạy với scaler/hoàn trả/dòng trùng và demo cuối vẫn đang chờ phần model của nhóm.
