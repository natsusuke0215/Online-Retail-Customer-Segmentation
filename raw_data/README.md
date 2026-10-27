# Dữ liệu gốc

Các file dữ liệu không được đưa lên Git. Để chạy baseline, tải **Online Retail** từ [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online%2Bretail) và đặt file Excel đúng tên:

```text
raw_data/Online Retail.xlsx
```

File đã dùng để tạo kết quả hiện tại có 541.909 dòng giao dịch, 8 cột và SHA-256:

```text
43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d
```

`online_retail_II.csv` là bản hai năm dùng cho phân tích độ nhạy, **không cần để chạy baseline** và hiện chưa được khôi phục trong workspace. Nếu muốn kiểm tra thêm lịch sử hai năm, tham khảo [Online Retail II trên UCI](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii); file CSV đã dùng trước đây có SHA-256 `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`. Không nối trực tiếp hai bộ dữ liệu vì chúng chồng lặp giao dịch.

UCI công bố hai bộ dữ liệu theo giấy phép **CC BY 4.0**. Khi sử dụng, ghi nguồn và tác giả Daqing Chen theo trang UCI tương ứng. Code trong repo này chưa được gán giấy phép riêng.
