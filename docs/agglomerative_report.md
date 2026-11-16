# BÁO CÁO TOÀN DIỆN VỀ PHÂN CỤM PHÂN CẤP TÍCH LŨY (AGGLOMERATIVE HIERARCHICAL CLUSTERING) TRONG BÀI TOÁN PHÂN KHÚC KHÁCH HÀNG BÁN LẺ

> **Dự án**: Online Retail Customer Segmentation  
> **Phương pháp**: Phân Cụm Phân Cấp Tích Lũy (Agglomerative Hierarchical Clustering)  
> **Dữ liệu**: Bộ dữ liệu giao dịch bán lẻ trực tuyến chuẩn hóa RFM (*Recency, Frequency, Monetary*)  
> **Mục tiêu**: Xây dựng mô hình phân khúc từ dưới lên (Bottom-up), phân tích cây phả hệ Dendrogram, thử nghiệm các phương pháp liên kết (Linkages), tìm số cụm tối ưu, phác họa chân dung khách hàng và kiểm chứng khả năng sinh lời trong 90 ngày tương lai.

---

## MỤC LỤC
1. [Giới Thiệu Bài Toán & Dữ Liệu RFM Bán Lẻ](#1-giới-thiệu-bài-toán--dữ-liệu-rfm-bán-lẻ)
2. [Bản Chất Của Phân Cụm Phân Cấp Tích Lũy (Agglomerative Clustering)](#2-bản-chất-của-phân-cụm-phân-cấp-tích-lũy-agglomerative-clustering)
3. [Khoảng Cách Không Gian & Các Tiêu Chí Liên Kết (Linkage Methods)](#3-khoảng-cách-không-gian--các-tiêu-chí-liên-kết-linkage-methods)
4. [Trực Quan Hóa Cây Phả Hệ Dendrogram & Phân Tích Điểm Cắt](#4-trực-quan-hóa-cây-phả-hệ-dendrogram--phân-tích-điểm-cắt)
5. [Thực Nghiệm Grid Search: Đánh Giá Đa Tiêu Chí với $k \in [2, 10]$](#5-thực-nghiệm-grid-search-đánh-giá-đa-tiêu-chí-với-k-in-2-10)
6. [Lựa Chọn Siêu Tham Số Tối Ưu & Biện Luận Khoa Học](#6-lựa-chọn-siêu-tham-số-tối-ưu--biện-luận-khoa-học)
7. [Chân Dung 4 Phân Khúc Khách Hàng (Customer Persona Profiling)](#7-chân-dung-4-phân-khúc-khách-hàng-customer-persona-profiling)
8. [Đánh Giá Ngoài Thời Gian (Out-of-Time Temporal Evaluation - 90 Ngày)](#8-đánh-giá-ngoài-thời-gian-out-of-time-temporal-evaluation---90-ngày)
9. [Chiến Lược Kinh Doanh & Kế Hoạch Marketing Thực Chiến](#9-chiến-lược-kinh-doanh--kế-hoạch-marketing-thực-chiến)
10. [Bộ Câu Hỏi Vấn Đáp (Q&A / Defense FAQ) Khi Bảo Vệ Đồ Án](#10-bộ-câu-hỏi-vấn-đáp-qa--defense-faq-khi-bảo-vệ-đồ-án)

---

## 1. Giới Thiệu Bài Toán & Dữ Liệu RFM Bán Lẻ

### 1.1. Bối Cảnh Thực Tế
Trong ngành thương mại điện tử và bán lẻ (Retail), doanh nghiệp sở hữu hàng ngàn đến hàng triệu khách hàng với hành vi mua sắm hoàn toàn khác nhau. Một doanh nghiệp không thể áp dụng cùng một chính sách tiếp thị cho tất cả khách hàng (đại trà / mass marketing) vì:
- Lãng phí ngân sách tiếp thị vào những khách hàng đã rời bỏ (churned).
- Bỏ lỡ cơ hội chăm sóc đặc quyền cho những khách hàng đem lại phần lớn doanh thu (VIPs/Whales).
- Làm phiền những khách hàng mới bằng các ưu đãi không phù hợp.

Do đó, **Phân khúc khách hàng (Customer Segmentation)** là kỹ thuật cốt lõi giúp chia tập khách hàng thành các nhóm có hành vi tương đồng, từ đó cá nhân hóa thông điệp tiếp thị và tối ưu hóa doanh thu.

### 1.2. Mô Hình Đặc Trưng RFM
Mô hình RFM là tiêu chuẩn vàng trong phân tích hành vi khách hàng bán lẻ, bao gồm 3 chỉ số then chốt tính trên mỗi khách hàng (`CustomerID`):
- **Recency ($R$) - Tính gần đây**: Số ngày tính từ lần phát sinh đơn hàng cuối cùng của khách hàng đến ngày cắt mốc phân tích (Snapshot date).  
  *Ý nghĩa*: Giá trị càng nhỏ chứng tỏ khách hàng mới mua gần đây, khả năng ghi nhớ thương hiệu và tiếp tục mua lại là rất cao.
- **Frequency ($F$) - Tần suất**: Tổng số lượng hóa đơn/đơn hàng hợp lệ mà khách hàng đã thực hiện trong quá khứ.  
  *Ý nghĩa*: Thể hiện mức độ gắn kết và lòng trung thành của khách hàng.
- **Monetary ($M$) - Giá trị tiền tệ**: Tổng số tiền thực tế (tính bằng bảng Anh £) mà khách hàng đã chi tiêu.  
  *Ý nghĩa*: Đóng góp trực tiếp vào doanh thu và lợi nhuận của doanh nghiệp.

### 1.3. Xử Lý Phân Phối Lệch (Skewness) & Chuẩn Hóa Đặc Trưng
Trong dữ liệu bán lẻ thực tế, các biến $F$ và $M$ bị hiện tượng **lệch phải cực nặng (heavy right-skewed)**: đại đa số khách hàng chỉ mua 1–2 lần với số tiền nhỏ (vài chục đến vài trăm bảng), trong khi có một số ít khách hàng bán buôn/doanh nghiệp mua hàng trăm lần với doanh số hàng chục ngàn bảng.

Nếu tính toán khoảng cách trực tiếp trên dữ liệu gốc:
1. Các khách hàng chi tiêu lớn sẽ áp đảo hoàn toàn khoảng cách Euclidean, khiến khoảng cách giữa các khách hàng bình thường bị triệt tiêu.
2. Đơn vị đo lường khác biệt (ngày vs số lần vs số tiền £) khiến biến Monetary có độ biến thiên lớn chi phối toàn bộ thuật toán.

**Quy trình tiền xử lý bắt buộc trong dự án:**
1. **Phép biến đổi Logarithm**: Áp dụng $x_{\text{log}} = \ln(1 + x)$ (`np.log1p`) lên cả $R$, $F$, $M$ để nén dải giá trị cực trị, đưa phân phối về gần dạng phân phối chuẩn (Gaussian-like).
2. **Chuẩn hóa Z-score (StandardScaler)**:
   $$z = \frac{x_{\text{log}} - \mu}{\sigma}$$
   Đưa 3 đặc trưng về cùng thang đo có trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$: `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled`.

> [!NOTE]
> Giai đoạn huấn luyện và lựa chọn mô hình (**Development Phase**) được thực hiện trên tập dữ liệu trước ngày **01/09/2011**, bao gồm **2.716 khách hàng**. Dữ liệu giai đoạn sau (từ 01/09/2011 đến 09/12/2011) được khóa hoàn toàn làm tập kiểm chứng tương lai 90 ngày.

---

## 2. Bản Chất Của Phân Cụm Phân Cấp Tích Lũy (Agglomerative Clustering)

### 2.1. Tiếp Cận "Từ Dưới Lên" (Bottom-Up Approach)
Phân cụm phân cấp tích lũy là thuật toán học không giám sát (Unsupervised Learning) hoạt động theo nguyên tắc gộp dần:
1. **Bước 0**: Ban đầu, mỗi điểm dữ liệu (mỗi khách hàng) được coi là một cụm riêng biệt gồm 1 phần tử. Nếu có $N$ khách hàng, ta có $N$ cụm ($N = 2.716$).
2. **Bước 1**: Tính toán ma trận khoảng cách giữa tất cả các cặp cụm hiện có.
3. **Bước 2**: Tìm hai cụm có khoảng cách/sai khác nhỏ nhất theo tiêu chí liên kết (Linkage) đã chọn và gộp chúng lại thành một cụm mới. Lúc này số lượng cụm giảm xuống còn $N - 1$.
4. **Lặp lại**: Lặp lại bước 2 và bước 3 liên tục $N - 1$ lần cho đến khi tất cả các điểm được gộp thành một cụm duy nhất chứa toàn bộ dữ liệu.

Toàn bộ quá trình gộp này được ghi lại dưới dạng một cấu trúc hình cây gọi là **Dendrogram** (Cây phả hệ).

```
   [Toàn bộ 2,716 khách hàng]  (1 Cụm lớn ở đỉnh)
              /        \
         [Cụm A]      [Cụm B]  (k = 2)
         /     \      /     \
      [C1]    [C2]  [C3]   [C4] (k = 4)
      ...     ...   ...    ...
     [x1]    [x2]  [x3]  ... [xN] (N Cụm đơn lẻ ban đầu)
```

### 2.2. So Sánh Với K-Means: Tại Sao Lại Dùng Agglomerative?

| Đặc Tính | K-Means Clustering | Agglomerative Hierarchical |
| :--- | :--- | :--- |
| **Yêu cầu số cụm $k$** | Bắt buộc phải chỉ định $k$ trước khi chạy. | **Không cần** chọn $k$ trước; xây dựng toàn bộ cây rồi mới cắt. |
| **Tính xác định (Determinism)** | Phụ thuộc khởi tạo tâm ngẫu nhiên (chạy lại có thể ra kết quả khác nhau). | **Hoàn toàn xác định 100%**: cùng dữ liệu và linkage luôn cho một kết quả duy nhất. |
| **Trực quan hóa cấu trúc** | Khó hình dung quan hệ cha-con giữa các phân khúc. | **Dendrogram** thể hiện trực quan cấu trúc phân cấp, quan hệ họ hàng giữa các nhóm. |
| **Dạng hình học của cụm** | Luôn giả định cụm hình cầu lồi (spherical) có kích thước bằng nhau. | Linh hoạt tùy thuộc vào tiêu chí liên kết (Ward, Complete, Average). |
| **Độ phức tạp tính toán** | Nhanh $O(N \cdot k \cdot I)$, phù hợp với hàng triệu bản ghi. | Chậm hơn, bộ nhớ $O(N^2)$, tính toán $O(N^2 \log N)$. Thích hợp cho tập khách hàng từ vài ngàn đến vài chục ngàn. |

---

## 3. Khoảng Cách Không Gian & Các Tiêu Chí Liên Kết (Linkage Methods)

Để gộp 2 cụm, thuật toán cần xác định: **Khoảng cách giữa hai cụm $A$ và $B$ được tính như thế nào?**  
Dưới đây là 4 phương pháp liên kết phổ biến trong khoa học dữ liệu:

```
    Linkage Types Illustration:
    
    Single:     min d(x, y)          o ---- o
    Complete:   max d(x, y)          o -------------- o
    Average:    mean d(x, y)         all pairs averaged
    Ward:       min variance ΔESS    centers & spread minimized
```

### 3.1. Ward's Linkage (`ward`) - Tối Thiểu Hóa Mức Tăng Phương Sai
- **Bản chất**: Ward không đo khoảng cách vật lý thông thường mà đo **mức độ gia tăng tổng bình phương sai số nội cụm ($\Delta ESS$)** khi gộp cụm $A$ và cụm $B$:
  $$\Delta ESS_{AB} = ESS_{A \cup B} - (ESS_A + ESS_B) = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$$
  Trong đó $n_A, n_B$ là số phần tử và $\mu_A, \mu_B$ là tâm hình học của hai cụm.
- **Ưu điểm**:
  - Tạo ra các cụm có kích thước tương đối cân bằng, hình cầu lồi (convex spherical).
  - Kháng nhiễu cực tốt, không bị hiện tượng một cụm nuốt trọn toàn bộ dữ liệu.
  - Phù hợp tuyệt đối với không gian khoảng cách Euclidean trên dữ liệu RFM đã chuẩn hóa z-score.
- **Hạn chế**: Chỉ áp dụng được với khoảng cách Euclidean.

### 3.2. Complete Linkage (`complete`) - Liên Kết Cực Đại
- **Bản chất**: Khoảng cách giữa 2 cụm là khoảng cách giữa **hai điểm xa nhất** thuộc 2 cụm:
  $$D(A, B) = \max_{x \in A, y \in B} d(x, y)$$
- **Ưu điểm**: Tạo ra các cụm rất chặt chẽ (compact), đường kính nội cụm nhỏ.
- **Hạn chế**: Rất nhạy cảm với các điểm ngoại lai (Outliers). Nếu có một khách hàng có giá trị cá biệt, khoảng cách xa nhất sẽ bị phóng đại khiến cụm bị méo mó.

### 3.3. Average Linkage (`average` / UPGMA) - Liên Kết Trung Bình
- **Bản chất**: Khoảng cách giữa 2 cụm là **trung bình cộng** khoảng cách giữa tất cả các cặp điểm:
  $$D(A, B) = \frac{1}{|A| \cdot |B|} \sum_{x \in A} \sum_{y \in B} d(x, y)$$
- **Ưu điểm**: Cân bằng dung hòa giữa Single và Complete, ít nhạy cảm với ngoại lai hơn Complete.
- **Hạn chế**: Khi dữ liệu có mật độ không đều, thuật toán có xu hướng gom các điểm dày đặc vào một cụm khổng lồ và tách từng điểm thưa thớt thành cụm đơn lẻ.

### 3.4. Single Linkage (`single`) - Liên Kết Cực Tiểu & Hiệu Ứng Xâu Chuỗi
- **Bản chất**: Khoảng cách giữa 2 cụm là khoảng cách giữa **hai điểm gần nhất**:
  $$D(A, B) = \min_{x \in A, y \in B} d(x, y)$$
- **Nhược điểm chí mạng**: Gặp hiện tượng **Xâu chuỗi (Chaining Effect)**. Chỉ cần có một chuỗi các điểm trung gian nằm giữa 2 cụm, thuật toán sẽ liên kết chúng thành một chuỗi dài dằng dặc, biến toàn bộ dữ liệu thành 1 cụm lớn duy nhất kèm vài điểm rác bên ngoài. Vì vậy, Single Linkage **hoàn toàn không phù hợp** cho bài toán phân khúc khách hàng.

---

## 4. Trực Quan Hóa Cây Phả Hệ Dendrogram & Phân Tích Điểm Cắt

### 4.1. Biểu Đồ So Sánh 4 Linkage
Khi chạy trên $2.716$ khách hàng thực tế, việc vẽ $2.716$ nhánh lá ở đáy sẽ khiến biểu đồ biến thành một mảng đen dày đặc. Vì vậy, ta áp dụng kỹ thuật chuẩn công nghiệp `truncate_mode='lastp'`, chỉ hiển thị $p = 30$ nút gộp cấp cao nhất (các số trong ngoặc đơn biểu thị số lượng khách hàng nằm trong nhánh con đó).

Biểu đồ so sánh chi tiết được lưu tại:  
[dendrogram_comparison.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/dendrogram_comparison.png)

```
                       DENDROGRAM SO SÁNH 4 PHƯƠNG PHÁP
      
           WARD (Cân bằng, rõ nét)              COMPLETE (Khá phân tán)
      Distance                             Distance
        100 |         /\                     10 |         /\
            |        /  \                       |        /  \
         50 |    /\      /\                   5 |     /\     /\
            |   /  \    /  \                    |    /  \   /  \
          0 +---()--()--()--()                0 +---()--()--()--()
            [Cụm đều, phân cấp đẹp]              [Bị kéo dài bởi outlier]
      
           AVERAGE (Mất cân đối)                SINGLE (Chaining Effect)
      Distance                             Distance
          6 |         /\                      3 |         /\
            |        /  \                       |        /  \
          3 |       /\   \                    1 |       /\   \______
            |      /  \   \                     |      /  \         \
          0 +-----()--()--()                  0 +-----()--()--------()
            [1 cụm ôm trọn 98% mẫu]              [Hiệu ứng xâu chuỗi dây thừng]
```

**Nhận xét từ thực nghiệm:**
1. **Single Linkage**: Biểu đồ dốc đứng một bên, các điểm cứ lần lượt bám vào một cụm chính duy nhất (hiệu ứng sợi mì/chaining).
2. **Average Linkage**: Nút gộp lớn nhất chứa đến $2.681$ khách hàng (chiếm 98,7% dữ liệu), chỉ có vài nhánh đơn độc tách ra.
3. **Ward Linkage**: Cấu trúc hình cây đối xứng và phân cấp rõ ràng nhất. Các nhánh chính tách nhau ở mức khoảng cách rất cao ($\Delta ESS > 30$), phản ánh các phân vùng khách hàng có tính tách biệt mạnh mẽ.

---

### 4.2. Phân Tích Điểm Cắt Ngang (Horizontal Cut Line) Trên Cây Ward
Biểu đồ chuyên sâu của Ward Linkage được lưu tại:  
[dendrogram_ward.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/dendrogram_ward.png)

```
         CÂY WARD LINKAGE & KHOẢNG NHẢY KHOẢNG CÁCH (MERGER GAP)
  Khoảng cách gộp (ΔESS)
    100 +                     /\
        |                    /  \
     80 |                   /    \
        |                  /      \
     60 |                 /        \
        |                /          \
     45 |               /            \
        |  - - - - - - / - - - - - - -\ - - - - - - - - - - - [Ngưỡng k=2]
        |             /\              /\
     38 | - - - - - -/ - - - - - - - / - - - - - - - - - - -  [Ngưỡng k=4: Distance = 38.0]
        |           /  \            /  \                      <--- KHOẢNG NHẢY LỚN (GAP)
     31 | - - - - -/- - \ - - - - -/- - \ - - - - - - - - -   [Ngưỡng k=5: Distance = 31.0]
        |         /\    /\        /\    /\
        |        /  \  /  \      /  \  /  \
      0 +-------()--()()--()----()--()()--()
               (215)  (319)    (980)  (1202)  <-- Số khách hàng mỗi cụm
```

**Nguyên tắc vàng khi chọn đường cắt trên Dendrogram:**
- Chúng ta tìm **đoạn thẳng đứng dài nhất (Longest Vertical Distance Gap)** mà không bị bất kỳ nhánh ngang nào cắt qua.
- Đoạn thẳng đứng này biểu diễn sự ổn định: khoảng cách phương sai tăng vọt trước khi hai cụm chịu gộp vào nhau.
- **Khoảng nhảy lớn nhất**: Nằm giữa khoảng cách $\sim 31.0$ và $\sim 45.0$.
  - Cắt tại **ngưỡng 38.0**: Ta cắt ngang qua 4 đường dọc, tạo ra đúng **$k = 4$ cụm** với quy mô cực kỳ cân đối: $215$, $319$, $980$, và $1.202$ khách hàng.
  - Cắt tại **ngưỡng 31.0**: Cho ra **$k = 5$ cụm**.

---

## 5. Thực Nghiệm Grid Search: Đánh Giá Đa Tiêu Chí với $k \in [2, 10]$

Chúng ta tiến hành chạy Grid Search đầy đủ cho 3 phương pháp liên kết (`ward`, `complete`, `average`) với số cụm $k$ chạy từ 2 đến 10 trên tập $2.716$ khách hàng.

Bảng kết quả được lưu tại:  
[linkage_comparison.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/tables/agglomerative/linkage_comparison.csv)  
Biểu đồ so sánh 3 chỉ số được lưu tại:  
[metrics_comparison.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/metrics_comparison.png)

### 5.1. Bảng Dữ Liệu Thực Nghiệm Chi Tiết

| Linkage | $k$ | Silhouette $\uparrow$ | Davies-Bouldin $\downarrow$ | Calinski-Harabasz $\uparrow$ | Cụm nhỏ nhất | Cụm lớn nhất | Thời gian ($s$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ward** | **2** | 0.3479 | 0.9904 | **1769.89** | 1.202 | 1.514 | 0.22 |
| **ward** | **3** | 0.3403 | **0.9854** | **1825.37** | 215 | 1.299 | 0.23 |
| **ward** | **4** | **0.3089** | **1.0440** | **1656.75** | **215** | **1.202** | **0.23** |
| **ward** | **5** | 0.3109 | 1.0775 | 1639.34 | 215 | 1.202 | 0.26 |
| **ward** | **6** | 0.2341 | 1.1244 | 1499.56 | 215 | 910 | 0.28 |
| **ward** | **7** | 0.2161 | 1.1202 | 1417.35 | 204 | 910 | 0.21 |
| **ward** | **8** | 0.2164 | 1.1598 | 1350.77 | 204 | 560 | 0.21 |
| complete | 2 | 0.4133 | 0.8190 | 812.61 | 198 | 2.518 | 0.23 |
| complete | 3 | 0.2620 | 1.3394 | 488.95 | 83 | 2.435 | 0.31 |
| complete | 4 | 0.2753 | 1.9172 | 584.62 | 83 | 2.340 | 0.24 |
| complete | 7 | 0.1683 | 1.0258 | 369.58 | **3** | 2.340 | 0.23 |
| complete | 9 | 0.1694 | 1.0318 | 489.40 | **1** | 1.603 | 0.27 |
| average | 2 | **0.5896** | 0.6249 | 328.51 | 35 | 2.681 | 0.25 |
| average | 3 | 0.4737 | 0.7450 | 176.38 | 5 | 2.681 | 0.32 |
| average | 4 | 0.3800 | 0.7279 | 125.23 | **4** | 2.677 | 0.22 |
| average | 5 | 0.3786 | 0.6534 | 95.10 | **1** | 2.677 | 0.27 |

---

### 5.2. Giải Thích Ý Nghĩa 3 Chỉ Số Đo Lường (Metrics)

#### A. Silhouette Score (Hệ số bóng đổ)
- **Công thức**: Với mỗi điểm $i$, gọi $a(i)$ là khoảng cách trung bình từ $i$ đến các điểm khác trong cùng cụm, $b(i)$ là khoảng cách trung bình từ $i$ đến các điểm thuộc cụm lân cận gần nhất:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s \in [-1, 1]$$
- **Ý nghĩa**:
  - $s \approx +1$: Điểm nằm rất sâu và chặt chẽ bên trong cụm của mình, cách rất xa cụm khác (hoàn hảo).
  - $s \approx 0$: Điểm nằm sát ranh giới giữa hai cụm (lưỡng lự).
  - $s < 0$: Điểm bị gán sai cụm.

#### B. Davies-Bouldin Index (Chỉ số Davies-Bouldin)
- **Công thức**:
  $$DB = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)} \right)$$
  Trong đó $\sigma_i$ là độ phân tán trung bình nội cụm $i$, và $d(\mu_i, \mu_j)$ là khoảng cách giữa 2 tâm cụm.
- **Ý nghĩa**: Đo độ tương đồng tồi nhất giữa các cặp cụm. **Càng nhỏ càng tốt** (cụm càng co cụm và càng cách xa nhau).

#### C. Calinski-Harabasz Index (Chỉ số tỷ số phương sai - Variance Ratio)
- **Công thức**:
  $$CH = \frac{\text{Trace}(B_k)}{\text{Trace}(W_k)} \times \frac{N - k}{k - 1}$$
  Trong đó $B_k$ là ma trận phân tán giữa các cụm (Between-group dispersion) và $W_k$ là ma trận phân tán trong nội bộ cụm (Within-group dispersion).
- **Ý nghĩa**: Tỷ số giữa phương sai ngoại cụm và phương sai nội cụm. **Càng lớn càng tốt**.

---

## 6. Lựa Chọn Siêu Tham Số Tối Ưu & Biện Luận Khoa Học

### 6.1. Cạm Bẫy Silhouette Của Average Linkage (Quan Trọng Khi Vấn Đáp!)
Nếu một người chỉ nhìn máy móc vào con số cao nhất, họ sẽ vội vã chọn `linkage='average'` tại $k=2$ vì có Silhouette lên đến **0.5896**.  
**Tuy nhiên, đây là một cạm bẫy toán học (Mathematical Artifact):**
- Nhìn vào cột kích thước cụm: Cụm 1 có **2.681 khách hàng**, cụm 2 chỉ có **35 khách hàng**.
- Khi $k=4$, `average` tạo ra các cụm với quy mô: $2.677$, $31$, $4$, và $4$ khách hàng!
- Thuật toán `average` chỉ đơn giản là cắt bỏ một vài điểm ngoại lai (outliers) nằm rải rác ngoài rìa và gom toàn bộ $99\%$ khách hàng còn lại vào một "nồi lẩu thập cẩm".
- Trong kinh doanh, việc tạo ra một phân khúc có 4 khách hàng là hoàn toàn **vô dụng và phi thực tế**.

> [!WARNING]
> Không bao giờ tối ưu hóa mù quáng một chỉ số đơn lẻ mà bỏ qua **sự phân bố kích thước cụm (Cluster Balance)** và **khả năng thực thi kinh doanh (Business Actionability)**!

### 6.2. Vì Sao Ward Linkage Với $k = 4$ Là Lựa Chọn Hoàn Hảo Nhất?
1. **Calinski-Harabasz vượt trội áp đảo**: Chỉ số $CH$ của Ward tại $k=4$ đạt **1.656,75**, cao gấp **3 lần Complete** ($584,62$) và gấp **13 lần Average** ($125,23$). Điều này chứng minh các cụm của Ward có mật độ nội cụm cô đặc nhất và tách biệt nhau rõ nét nhất trong không gian Euclidean.
2. **Kích thước cụm cực kỳ cân đối**:
   - Cụm 2: $215$ khách hàng ($7,9\%$).
   - Cụm 3: $319$ khách hàng ($11,7\%$).
   - Cụm 0: $980$ khách hàng ($36,1\%$).
   - Cụm 1: $1.202$ khách hàng ($44,3\%$).
   Không có cụm nào bị cô lập thành thiểu số ($<1\%$).
3. **Phù hợp hoàn hảo với lý thuyết phân khúc bán lẻ**:
   Bộ 4 nhóm tương ứng trọn vẹn với 4 góc độ hành vi khách hàng cốt lõi: *Khách hàng VIP*, *Khách hàng trung thành*, *Khách hàng tiềm năng*, và *Khách hàng có nguy cơ rời bỏ*.

---

## 7. Chân Dung 4 Phân Khúc Khách Hàng (Customer Persona Profiling)

Sau khi huấn luyện mô hình tối ưu `AgglomerativeClustering(n_clusters=4, linkage='ward')`, chúng ta kết nối nhãn cụm với bảng dữ liệu RFM gốc để tính các chỉ số thống kê thực tế.

Bảng hồ sơ phân khúc được lưu tại:  
[cluster_profiles.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/tables/agglomerative/cluster_profiles.csv)  
Bản đồ phân cụm PCA 2D:  
[pca_clusters.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/pca_clusters.png)  
Biểu đồ phân phối RFM (Boxplots):  
[rfm_distribution.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/rfm_distribution.png)  
Biểu đồ so sánh giá trị trung vị:  
[cluster_profiles.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/cluster_profiles.png)

### 7.1. Bảng Tổng Hợp Đặc Trưng RFM Thực Tế

| Cụm | Tên Định Danh Chân Dung (Persona) | Số KH | Tỷ Lệ KH | Median $R$ (ngày) | Median $F$ (đơn) | Median $M$ (£) | Tổng Doanh Thu (£) | Tỷ Trọng Doanh Thu |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | **Champions / VIPs (Chi tiêu khủng, gắn kết cao)** | **215** | **7,92%** | **9,0** | **9,0** | **3.450,58 £** | **1.488.393,45 £** | **45,56%** |
| **0** | **Loyal Customers (Khách hàng thân thiết)** | **980** | **36,08%** | **27,0** | **3,0** | **924,57 £** | **1.261.410,06 £** | **38,61%** |
| **3** | **Potential / Promising (Khách mới / Đầy triển vọng)** | **319** | **11,75%** | **20,0** | **1,0** | **304,30 £** | **111.402,69 £** | **3,41%** |
| **1** | **At-Risk / Lost (Nguy cơ rời bỏ / Đã mất)** | **1.202** | **44,26%** | **97,0** | **1,0** | **261,16 £** | **405.782,08 £** | **12,42%** |

```
                       MINH HỌA QUY TẮC PARETO TRONG DỰ ÁN
        
          TỶ LỆ KHÁCH HÀNG                          ĐÓNG GÓP DOANH THU
     [ VIP ] 7.9%                             [ VIP ] 45.6%
     [ Loyal ] 36.1%                          [ Loyal ] 38.6%
     [ Potential ] 11.7%                      [ Potential ] 3.4%
     [ At-Risk ] 44.3%                        [ At-Risk ] 12.4%
     
     --> VIP + Loyal = 44% Khách hàng  =======>  Đóng góp 84.2% Doanh thu!
```

---

### 7.2. Phân Tích Sâu Sắc Từng Nhóm Khách Hàng

#### 🌟 Cụm 2: Champions / VIPs (Hạng Tinh Anh)
- **Đặc trưng số liệu**:
  - Mua hàng cực kỳ gần đây: Trung vị Recency chỉ **9 ngày** (trung bình 12 ngày).
  - Tần suất mua hàng vượt trội: Trung vị **9 đơn hàng** (trung bình gần 12 đơn).
  - Chi tiêu khổng lồ: Trung vị **3.450,58 £** (trung bình lên tới **6.922,76 £**).
- **Giá trị kinh doanh**: Chỉ chiếm vỏn vẹn **7,92% khách hàng** nhưng đem về tới **45,56% tổng doanh số** của toàn bộ doanh nghiệp! Đây chính là các khách hàng sỉ, nhà phân phối hoặc cá nhân trung thành bậc nhất.

#### 🤝 Cụm 0: Loyal Customers (Khách Hàng Thân Thiết)
- **Đặc trưng số liệu**:
  - Recency trung vị **27 ngày** (dưới 1 tháng).
  - Tần suất trung vị **3 đơn hàng**.
  - Chi tiêu trung vị **924,57 £** (trung bình **1.287,15 £**).
- **Giá trị kinh doanh**: Chiếm **36,08% lượng khách** và đóng góp **38,61% doanh thu**. Họ là xương sống duy trì dòng tiền ổn định hàng tháng cho doanh nghiệp.

#### 🌱 Cụm 3: Potential / Promising (Khách Hàng Mới & Triển Vọng)
- **Đặc trưng số liệu**:
  - Mua hàng khá gần đây: Recency trung vị **20 ngày**.
  - Tần suất mua: Mới chỉ mua **1 đơn hàng**.
  - Chi tiêu trung vị: **304,30 £**.
- **Giá trị kinh doanh**: Đây là những khách hàng mới trải nghiệm mua sắm trong vài tuần gần đây. Họ có tiềm năng trở thành khách hàng thân thiết nếu được chăm sóc đúng cách.

#### ⚠️ Cụm 1: At-Risk / Lost (Khách Hàng Ngủ Đông & Nguy Cơ Rời Bỏ)
- **Đặc trưng số liệu**:
  - Đã rất lâu không quay lại: Recency trung vị **97 ngày** (hơn 3 tháng không mua sắm).
  - Tần suất rất thấp: Trung vị **1 đơn hàng**.
  - Chi tiêu thấp: Trung vị **261,16 £**.
- **Giá trị kinh doanh**: Chiếm tới **44,26% tổng số khách**, nhưng chỉ đóng góp **12,42% doanh thu**. Nhiều khách hàng trong số này đã rời bỏ thương hiệu sang đối thủ cạnh tranh.

---

## 8. Đánh Giá Ngoài Thời Gian (Out-of-Time Temporal Evaluation - 90 Ngày)

### 8.1. Bản Chất Đánh Giá Ngoài Thời Gian
Hầu hết các bài toán phân cụm thông thường dừng lại ở việc tính Silhouette score trên dữ liệu quá khứ. Tuy nhiên, trong thực tế, giá trị tối cao của mô hình là: **Nhãn phân cụm quá khứ có dự báo chính xác hành vi tương lai hay không?**

Dự án thiết lập một bài kiểm tra nghiêm ngặt: Lấy nhãn cụm được huấn luyện từ dữ liệu trước ngày 01/09/2011, sau đó theo dõi hành vi mua sắm thực tế của các khách hàng này trong **90 ngày tiếp theo** (từ 01/09/2011 đến 09/12/2011).

Báo cáo kết quả được lưu tại:  
[future_cluster_profiles.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/temporal/development/future_cluster_profiles.csv)  
[cluster_policy_comparison.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/temporal/development/cluster_policy_comparison.csv)

### 8.2. Kết Quả Theo Dõi 90 Ngày Tương Lai (Giai Đoạn Development)

| Cụm | Nhóm Khách Hàng | Khách Hàng | Số Người Mua Lại 90 Ngày | **Tỷ Lệ Mua Lại (Repurchase Rate)** | Doanh Thu 90 Ngày Tới (£) | Doanh Thu TB/Người (£) | **Tỷ Trọng Doanh Thu Tương Lai** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | **Champions / VIP** | 215 | 204 | **94,88%** | **837.740,23 £** | **3.896,47 £** | **54,74%** |
| **0** | **Loyal Customers** | 980 | 633 | **64,59%** | **437.813,65 £** | **446,75 £** | **28,61%** |
| **3** | **Potential** | 319 | 131 | **41,07%** | **52.296,52 £** | **163,94 £** | **3,42%** |
| **1** | **At-Risk / Lost** | 1.202 | 369 | **30,70%** | **202.536,79 £** | **168,50 £** | **13,23%** |

```
                HÀNH VI MUA HÀNG 90 NGÀY TƯƠNG LAI CỦA CÁC CỤM
                
      TỶ LỆ QUAY LẠI MUA HÀNG (%)          DOANH THU ĐÓNG GÓP TƯƠNG LAI (£)
  VIP       |==================== 94.9%    VIP       |================= 837.7k£ (54.7%)
  Loyal     |============= 64.6%           Loyal     |========= 437.8k£ (28.6%)
  Potential |======== 41.1%                At-Risk   |==== 202.5k£ (13.2%)
  At-Risk   |====== 30.7%                  Potential |= 52.3k£ (3.4%)
```

### 8.3. Những Bằng Chứng Thực Nghiệm Vô Giá
1. **Khả năng dự báo hành vi mua lại chuẩn xác đến kinh ngạc**:
   - Nhóm VIP (Cụm 2) có tỷ lệ quay lại mua hàng đạt **94,88%** (204 trên tổng số 215 người tiếp tục mua hàng trong 90 ngày tới!).
   - Nhóm Loyal đạt **64,59%**.
   - Nhóm At-Risk tụt dốc xuống chỉ còn **30,70%**.
   Thứ tự tỷ lệ mua lại phản ánh chính xác 100% thứ bậc sức khỏe khách hàng mà mô hình đã phân khúc.
2. **Sức mạnh doanh thu của nhóm VIP**:
   Trong 90 ngày tương lai, chỉ 215 khách hàng VIP đã mang về tới **837.740,23 £**, chiếm **54,74% tổng doanh thu tương lai** của toàn bộ hệ thống! Trung bình mỗi khách hàng VIP chi tiêu gần **3.900 £** trong 3 tháng.
3. **Kiểm chứng giai đoạn Final Phase (3.314 khách hàng)**:
   Khi áp dụng mô hình lên giai đoạn Final (dữ liệu lớn hơn), kết quả hoàn toàn nhất quán:
   - Nhóm VIP tiếp tục đạt tỷ lệ mua lại **96,73%** (266 / 275 khách hàng quay lại).
   - Đóng góp **1.307.326,81 £** (chiếm **52,40%** doanh thu).
   - Doanh thu trung bình mỗi khách VIP đạt **4.753,92 £**.

---

## 9. Chiến Lược Kinh Doanh & Kế Hoạch Marketing Thực Chiến

Dựa trên kết quả phân cụm Agglomerative, doanh nghiệp có thể triển khai kế hoạch hành động chi tiết (Marketing Playbook) cho từng phân khúc:

```
               CHIẾN LƯỢC KINH DOANH CHO TỪNG PHÂN KHÚC
   
   [ CỤM 2: CHAMPIONS / VIP ]      [ CỤM 0: LOYAL CUSTOMERS ]
   - Chăm sóc 1:1 đặc quyền        - Chương trình tích điểm VIP
   - Trải nghiệm sớm sản phẩm      - Bán kèm & Nâng hạng (Up-sell)
   - Tri ân quà sinh nhật/lễ       - Ưu đãi freeship định kỳ
   
   [ CỤM 3: POTENTIAL / NEW ]      [ CỤM 1: AT-RISK / LOST ]
   - Kích hoạt đơn hàng thứ 2      - Email Win-back "We miss you"
   - Giảm giá 15% lần mua kế       - Khuyến mãi sâu có thời hạn
   - Khảo sát sự hài lòng          - Lọc bỏ nếu không phản hồi
```

### 9.1. Phân Khúc VIP (Cụm 2) - Chiến Lược "Bảo Vệ & Đặc Quyền"
- **Mục tiêu**: Tối đa hóa trải nghiệm, ngăn chặn tuyệt đối việc mất khách hàng vào tay đối thủ.
- **Hành động cụ thể**:
  - Cung cấp nhân viên quản lý tài khoản riêng (Dedicated Account Manager / Hotline VIP).
  - Quyền truy cập độc quyền: Cho phép đặt mua trước các bộ sưu tập mới (Early Access) trước 48 giờ.
  - Quà tặng tri ân cá nhân hóa nhân dịp sinh nhật hoặc lễ tết (rượu vang, thiệp viết tay).
  - *Lưu ý*: Tuyệt đối không gửi các email giảm giá đại trà, làm giảm hình ảnh thương hiệu trong mắt nhóm khách này.

### 9.2. Phân Khúc Thân Thiết (Cụm 0) - Chiến Lược "Gia Tăng Giá Trị & Nâng Hạng"
- **Mục tiêu**: Thúc đẩy tần suất mua hàng và nâng giá trị đơn hàng trung bình (AOV) để biến họ thành VIP.
- **Hành động cụ thể**:
  - Thiết lập chương trình khách hàng thân thiết (Loyalty Tier): Tích điểm đổi voucher cho các mốc đơn hàng cao.
  - Thuật toán gợi ý sản phẩm liên quan (Cross-selling / Up-selling) dựa trên giỏ hàng thường mua.
  - Miễn phí vận chuyển (Free Shipping) cho đơn hàng đạt giá trị tối thiểu từ 500 £ trở lên.

### 9.3. Phân Khúc Tiềm Năng (Cụm 3) - Chiến Lược "Kích Hoạt Đơn Hàng Thứ Hai"
- **Mục tiêu**: Rút ngắn thời gian chuyển đổi từ người mua lần đầu thành khách hàng thường xuyên.
- **Hành động cụ thể**:
  - Gửi chuỗi email Onboarding chào mừng kèm mã giảm giá 10%–15% áp dụng trong vòng 14–30 ngày.
  - Gửi tin nhắn khảo sát ngắn: "Bạn thấy trải nghiệm nhận hàng vừa qua thế nào?" để giải quyết khiếu nại kịp thời.

### 9.4. Phân Khúc Nguy Cơ Rời Bỏ (Cụm 1) - Chiến Lược "Tái Kích Hoạt Có Chọn Lọc"
- **Mục tiêu**: Kéo khách hàng quay lại với chi phí thấp nhất; thanh lọc tệp khách hàng không còn giá trị.
- **Hành động cụ thể**:
  - Chiến dịch tái kích hoạt (Win-back Campaign): Gửi thông điệp "Chúng tôi nhớ bạn" kèm mã Flash Sale giảm giá sâu trong 3 ngày.
  - Sau 2 lần gửi chiến dịch tái kích hoạt mà khách hàng vẫn không mở email hoặc không mua hàng, hãy **loại bỏ họ khỏi danh sách gửi email tự động hàng tuần** để giảm chi phí phần mềm tiếp thị và cải thiện tỷ lệ vào hòm thư chính (Deliverability).

---

## 10. Bộ Câu Hỏi Vấn Đáp (Q&A / Defense FAQ) Khi Bảo Vệ Đồ Án

Bộ câu hỏi và câu trả lời mẫu này giúp bạn tự tin trả lời xuất sắc mọi câu hỏi phản biện của giảng viên và hội đồng chấm thi:

#### Câu 1: Tại sao em lại chọn Agglomerative Clustering thay vì K-Means cho bài toán này?
> **Trả lời mẫu**:  
> *"Dạ thưa thầy/cô, K-Means tuy tính toán nhanh nhưng có nhược điểm là kết quả phụ thuộc vào việc khởi tạo tâm ngẫu nhiên và ta buộc phải đoán định trước số cụm $k$. Trong khi đó, Agglomerative Hierarchical Clustering là thuật toán có tính xác định hoàn toàn (100% deterministic), giúp chúng em quan sát được toàn bộ cây phả hệ phân cấp (Dendrogram) từ mức độ chi tiết đến tổng quát. Nhờ cấu trúc hình cây này, chúng em có thể phân tích khoảng nhảy phương sai và xác định được điểm cắt tự nhiên của dữ liệu khách hàng mà K-Means không thể hiện được."*

#### Câu 2: Em hãy giải thích bản chất toán học của Ward Linkage và tại sao nó lại vượt trội hơn Complete hay Average?
> **Trả lời mẫu**:  
> *"Dạ, Ward Linkage hoạt động dựa trên nguyên lý tối thiểu hóa mức tăng tổng bình phương sai số nội cụm ($\Delta ESS$) ở mỗi bước gộp. Về mặt hình học, Ward ưu tiên tạo ra các cụm hình cầu lồi có kích thước đồng đều và mật độ chặt chẽ. Trong thực nghiệm của chúng em:  
> - Average Linkage bị cạm bẫy cô lập ngoại lai: nó gom 99% khách hàng vào một cụm và tạo ra các cụm chỉ có 1–4 khách hàng, hoàn toàn mất đi ý nghĩa kinh doanh.  
> - Ward Linkage tạo ra 4 cụm cực kỳ cân đối, đồng thời đạt chỉ số Calinski-Harabasz lên tới 1.656, cao gấp 3 lần Complete và 13 lần Average, chứng minh các cụm được phân tách rõ rệt nhất trong không gian Euclidean."*

#### Câu 3: Làm thế nào em xác định được số cụm $k = 4$ từ Dendrogram?
> **Trả lời mẫu**:  
> *"Dạ, trên trục tung của Dendrogram Ward (biểu thị mức gia tăng phương sai), nguyên tắc là tìm khoảng trống thẳng đứng dài nhất (Largest Vertical Gap) không bị nhánh gộp ngang nào cắt qua. Chúng em quan sát thấy khoảng cách từ 31 đến 45 là khoảng nhảy lớn nhất. Cắt ngang tại ngưỡng khoảng cách 38.0 cho ra đúng $k = 4$ cụm. Số cụm này vừa đạt chỉ số Silhouette tốt ($0,3089$), vừa khớp hoàn hảo với 4 nhóm hành vi kinh điển trong bán lẻ: Champions, Loyal, Potential, và At-Risk."*

#### Câu 4: Vì sao dữ liệu RFM bắt buộc phải log-transform và Standard Scaler trước khi phân cụm?
> **Trả lời mẫu**:  
> *"Dạ, trong bán lẻ, biến Frequency và Monetary bị lệch phải cực nặng do sự xuất hiện của các khách hàng chi tiêu lớn (Whales). Nếu không log-transform (`log1p`), một vài khách hàng chi hàng chục ngàn bảng sẽ kéo giãn khoảng cách Euclidean và làm lu mờ sự khác biệt giữa hàng ngàn khách hàng bình thường khác. Phép biến đổi log giúp kéo phân phối về dạng chuẩn, và `StandardScaler` đưa 3 biến về cùng thang đo trung bình bằng 0, độ lệch chuẩn bằng 1, giúp thuật toán đánh giá công bằng cả 3 khía cạnh R, F và M."*

#### Câu 5: Đánh giá ngoài thời gian (Out-of-time Temporal Evaluation) có ý nghĩa gì và kết quả ra sao?
> **Trả lời mẫu**:  
> *"Dạ, nếu chỉ đánh giá bằng Silhouette score trên dữ liệu quá khứ thì mô hình có thể bị overfitting hoặc không có giá trị thực tiễn. Chúng em đã theo dõi hành vi của các cụm này trong 90 ngày tương lai và phát hiện:  
> - Cụm VIP có tỷ lệ quay lại mua hàng đạt **94,88%**, đóng góp tới **54,74% tổng doanh thu tương lai**.  
> - Trong khi đó cụm At-Risk chỉ có **30,70%** mua lại.  
> Kết quả này chứng minh nhãn phân cụm của mô hình có khả năng dự báo cực kỳ chuẩn xác hành vi mua sắm thực tế của khách hàng trong tương lai."*

---

## TỔNG KẾT & DANH MỤC TÀI NGUYÊN DỰ ÁN

| Tài Nguyên | Đường Dẫn Tệp | Mô Tả |
| :--- | :--- | :--- |
| **Mã nguồn thực thi** | [agglomerative.py](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/agglomerative.py) | Script pipeline tự động chạy toàn bộ quy trình từ dữ liệu thô đến xuất kết quả |
| **Jupyter Notebook** | [agglomerative.ipynb](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/agglomerative.ipynb) | Notebook tương tác trực quan từng bước |
| **Bảng so sánh Grid Search** | [linkage_comparison.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/tables/agglomerative/linkage_comparison.csv) | Bảng dữ liệu thực nghiệm 27 mô hình (Ward, Complete, Average qua $k \in [2, 10]$) |
| **Hồ sơ phân khúc KH** | [cluster_profiles.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/tables/agglomerative/cluster_profiles.csv) | Bảng thống kê Mean/Median RFM và doanh thu của 4 cụm |
| **Đánh giá ngoài thời gian** | [future_cluster_profiles.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/temporal/development/future_cluster_profiles.csv) | Bảng theo dõi tỷ lệ mua lại và doanh thu 90 ngày tương lai |
| **Dendrogram so sánh** | [dendrogram_comparison.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/dendrogram_comparison.png) | Hình ảnh so sánh cây phân cấp giữa 4 linkages |
| **Dendrogram Ward** | [dendrogram_ward.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/dendrogram_ward.png) | Hình ảnh phân tích đường cắt ngang và khoảng nhảy khoảng cách |
| **Biểu đồ đa tiêu chí** | [metrics_comparison.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/metrics_comparison.png) | Đồ thị Silhouette, Davies-Bouldin, Calinski-Harabasz theo $k$ |
| **Bản đồ cụm PCA 2D** | [pca_clusters.png](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/reports/figures/agglomerative/pca_clusters.png) | Trực quan hóa không gian phân tách khách hàng trên 2 thành phần chính |
| **File nhãn nộp bài (Dev)** | [agglomerative.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/models/outputs/development/agglomerative.csv) | Nhãn phân cụm 2.716 khách hàng giai đoạn development |
| **Metadata nộp bài (Dev)** | [agglomerative.json](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/models/outputs/development/agglomerative.json) | Siêu dữ liệu cấu hình tham số và chỉ số đo lường |
| **File nhãn nộp bài (Final)** | [agglomerative.csv](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/models/outputs/final/agglomerative.csv) | Nhãn phân cụm 3.314 khách hàng giai đoạn final |
| **Metadata nộp bài (Final)** | [agglomerative.json](file:///D:/WORKSPACE/Online-Retail-Customer-Segmentation/models/outputs/final/agglomerative.json) | Siêu dữ liệu cấu hình tham số và chỉ số đo lường giai đoạn final |
