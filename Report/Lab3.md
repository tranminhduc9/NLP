# Lab 3: Trực quan hóa Word Embeddings

---

## Mục tiêu
- Sử dụng kỹ thuật **giảm chiều** (PCA, t-SNE) để giảm word vectors từ không gian nhiều chiều (100D) xuống 2D
- **Trực quan hóa** 5,000 từ trong không gian 2D để quan sát mối quan hệ ngữ nghĩa giữa các từ
- **So sánh** hiệu quả của PCA và t-SNE trong việc trực quan hóa word embeddings
- **Phân tích** các nhóm từ theo chủ đề để kiểm chứng mô hình học được semantic similarity
- **Tìm kiếm** và trực quan hóa các từ gần nghĩa

---

## Nhiệm vụ

### Task 1: Cài đặt thư viện và tải mô hình Word Embeddings

**Thư viện cần thiết:**
- **gensim**: Thư viện NLP, hỗ trợ tải và làm việc với word embeddings pre-trained
- **scikit-learn**: Cung cấp các thuật toán giảm chiều như PCA, t-SNE
- **matplotlib**: Thư viện vẽ đồ thị 2D/3D để trực quan hóa dữ liệu
- **numpy**: Xử lý ma trận và vector hiệu quả

**Mô hình GloVe:**
- **GloVe** (Global Vectors for Word Representation) là mô hình word embedding pre-trained
- Được huấn luyện trên **Wikipedia + Gigaword corpus** (hàng tỷ từ)
- Mỗi từ được biểu diễn bằng **vector 100 chiều**
- **Ưu điểm**: Nhanh, chất lượng cao, không cần huấn luyện lại, vocabulary lớn (400K từ)


**Load model:**
```python
import gensim.downloader as api
model = api.load('glove-wiki-gigaword-100')
```

---

### Task 2: Chọn 5,000 từ để trực quan hóa

**Kích thước dữ liệu:**
- **Input**: 5,000 từ × 100 chiều = Ma trận 5000×100
- **Output** sau giảm chiều: 5000×2 (để vẽ biểu đồ 2D)

---

### Task 3: Giảm chiều với PCA

**Về PCA (Principal Component Analysis):**

**Nguyên lý hoạt động:**
1. PCA tìm các **trục chính** (principal components) của dữ liệu
2. **Trục PC1**: Hướng có phương sai lớn nhất
3. **Trục PC2**: Hướng vuông góc với PC1, có phương sai lớn thứ hai
4. Giảm 100 chiều → 2 chiều bằng cách **chiếu** các điểm lên 2 trục này

**Ưu điểm:**
-  **Nhanh** và hiệu quả: O(n²d) với n=5000 từ, d=100 chiều
-  **Bảo toàn cấu trúc toàn cục**: Từ xa nhau trong 100D vẫn xa trong 2D
-  **Kết quả ổn định**: Deterministic với cùng random_state
-  Phù hợp làm **preprocessing** cho các thuật toán khác

**Nhược điểm:**
-  **Tuyến tính** → không bắt được quan hệ phi tuyến
-  **Mất nhiều thông tin**: Chỉ giữ ~10-15% phương sai với 2 chiều
-  **Cluster mờ nhạt**: Khó phân biệt các nhóm từ rõ ràng

**Phân tích Explained Variance:**
- Tỷ lệ phương sai được giữ lại thường ~10-15% với 2 chiều
- 85-90% thông tin bị mất do hạn chế của không gian 2D
- Tuy nhiên vẫn đủ để quan sát các mối quan hệ cơ bản

---

### Task 4: Trực quan hóa với PCA (2D)

**Chiến lược visualization:**
- Vẽ tất cả **5,000 điểm** (màu xanh, alpha thấp để tránh đè lấn)
- Hiển thị **label** cho 200 từ phổ biến nhất để tránh lộn xộn
- Sử dụng `plt.annotate()` để gắn nhãn

**Phân tích biểu đồ:**

**Quan sát:**
- Các từ **phân tán đều** trong không gian 2D
- Có sự **phân cụm nhẹ** nhưng không rõ ràng
- **Ranh giới** giữa các nhóm từ mờ nhạt

**Giải thích:**
- PCA là phương pháp **tuyến tính** → không tách rõ các cluster phi tuyến
- Bảo toàn khoảng cách toàn cục → từ xa nhau trong 100D vẫn xa trong 2D
- Phù hợp để có **cái nhìn tổng quan** về phân bố dữ liệu

---

### Task 5: Giảm chiều với t-SNE

**Về t-SNE (t-Distributed Stochastic Neighbor Embedding):**

**Nguyên lý hoạt động:**
1. Tính **xác suất tương đồng** giữa các cặp điểm trong không gian 100D
2. Khởi tạo **ngẫu nhiên** vị trí trong không gian 2D
3. **Tối ưu hóa** để xác suất trong 2D giống với xác suất trong 100D
4. Sử dụng **phân phối Student's t** để tránh "crowding problem"

**Ưu điểm:**
-  Bảo toàn cấu trúc **cục bộ** rất tốt
-  Tạo ra các **cluster rõ ràng**, dễ quan sát
-  Phát hiện được **quan hệ phi tuyến** phức tạp
-  Tốt cho **trực quan hóa** và presentation

**Nhược điểm:**
-  **Chậm** O(n² log n) - với 5000 từ mất 1-3 phút
-  **Không bảo toàn khoảng cách toàn cục**
-  **Stochastic** - kết quả thay đổi mỗi lần chạy (dù có random_state)
-  Khó **diễn giải** trục tọa độ (không như PC1, PC2 của PCA)

**Tham số quan trọng:**
- `perplexity=30`: Cân bằng giữa cấu trúc local và global (thường 5-50)
- `max_iter=1000`: Số vòng lặp tối ưu hóa
- `random_state=42`: Đảm bảo kết quả có thể tái tạo

---

### Task 6: Trực quan hóa với t-SNE (2D)

**Phân tích biểu đồ:**

**Quan sát:**
- Các từ tạo thành **các cluster rõ ràng**, tách biệt hơn PCA nhiều
- Từ **cùng ngữ nghĩa/chủ đề** nhóm lại gần nhau
- Có **khoảng trống** giữa các cluster

**So sánh với PCA:**

| Tiêu chí | PCA | t-SNE |
|----------|-----|-------|
| **Cluster** | Mờ nhạt | Rõ ràng |
| **Phân tách** | Liên tục | Có khoảng trống |
| **Cấu trúc cục bộ** | Yếu | Mạnh |
| **Tốc độ** | Nhanh | Chậm |
| **Hiệu quả cho NLP** | Trung bình | Tốt |

**Ứng dụng:**
- t-SNE phù hợp để **khám phá** cấu trúc ngữ nghĩa
- Giúp **trực quan hóa** mối quan hệ giữa các từ
- Hữu ích cho **phân tích định tính** và presentation

---

### Task 7: So sánh PCA và t-SNE

**Mục đích:**
Đặt hai biểu đồ cạnh nhau để **so sánh trực quan** sự khác biệt giữa hai phương pháp

**Điểm khác biệt rõ nhất:**

**1. Độ phân tán:**
- **PCA**: Từ phân bố đều, không có ranh giới rõ
- **t-SNE**: Từ tập trung thành từng đám, có khoảng trống

**2. Bảo toàn thông tin:**
- **PCA**: Bảo toàn khoảng cách toàn cục (từ xa nhau vẫn xa)
- **t-SNE**: Bảo toàn láng giềng gần (từ gần nhau vẫn gần)

**3. Khả năng phân nhóm:**
- **PCA**: Khó phân biệt các nhóm từ
- **t-SNE**: Các nhóm từ tách biệt rõ ràng

**Khi nào dùng phương pháp nào?**

**Dùng PCA khi:**
-  Cần tốc độ nhanh
-  Dữ liệu lớn (>100K điểm)
-  Cần bảo toàn khoảng cách tuyệt đối
-  Cần kết quả ổn định, reproducible
-  Làm preprocessing cho thuật toán khác

**Dùng t-SNE khi:**
-  Cần trực quan hóa cluster rõ ràng
-  Khám phá cấu trúc dữ liệu
-  Dữ liệu vừa và nhỏ (<50K điểm)
-  Chấp nhận thời gian xử lý lâu hơn
-  Mục đích presentation/demo

---

### Task 8: Trực quan hóa nhóm từ cụ thể

**Mục đích thí nghiệm:**
Kiểm chứng xem mô hình word embedding có **học được mối quan hệ ngữ nghĩa** hay không bằng cách quan sát các nhóm từ cùng chủ đề

**Các nhóm từ được chọn:**
1. **Quốc gia**: america, china, japan, france, germany, russia, india, brazil
2. **Thủ đô**: washington, beijing, tokyo, paris, berlin, moscow, delhi, london
3. **Động vật**: dog, cat, lion, tiger, elephant, monkey, bird, fish
4. **Màu sắc**: red, blue, green, yellow, black, white, orange, purple
5. **Số**: one, two, three, four, five, six, seven, eight, nine, ten

**Giả thuyết:**
- Nếu model tốt → các từ **cùng nhóm nên ở gần nhau** trong không gian vector
- Các nhóm khác nhau nên **tách biệt nhau**
- Quốc gia và thủ đô có thể gần nhau (cùng domain địa lý)

**Kỹ thuật trực quan hóa:**
- Mỗi nhóm có **màu riêng** để phân biệt
- Từ trong nhóm được **highlight** với marker lớn

**Phân tích kết quả:**

**1. t-SNE thể hiện rõ hơn PCA:**
- Các từ cùng nhóm **tập trung gần nhau** rõ ràng hơn
- Ví dụ: Các số (one, two, three...) tạo thành cluster riêng biệt
- Các quốc gia và thủ đô thường ở gần nhau (cùng domain)

**2. Mối quan hệ ngữ nghĩa được học:**
- Quốc gia và thủ đô có xu hướng gần nhau 
- Động vật tập trung riêng
- Màu sắc có thể phân tán hơn

---

### Task 9: Tìm từ gần nghĩa của `computer`

**Mục đích:**
Minh họa ứng dụng thực tế của word embeddings: **Tìm từ đồng nghĩa/gần nghĩa**

**Phương pháp:**
- Sử dụng **cosine similarity** để đo độ tương đồng giữa vectors
- Công thức: `similarity(A, B) = (A · B) / (||A|| × ||B||)`
- Giá trị: -1 (ngược nghĩa) → 0 (không liên quan) → 1 (giống nhau)


**Trực quan hóa:**
- **Plot 1 (Overview)**: Hiển thị toàn bộ 5,000 từ với highlighting
  - Background: màu xám nhạt
  - Similar words: màu xanh dương (marker tròn)
  - Target word 'computer': màu đỏ (marker sao lớn)
  
- **Plot 2 (Zoom)**: Phóng to khu vực xung quanh 'computer'
  - Dễ quan sát **mối quan hệ không gian** giữa các từ
  - Font size lớn hơn để dễ đọc
  - Bounding box tự động tính dựa trên vị trí các từ
---

## Cách chạy

### 1. Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

### 2. Chạy Jupyter Notebook:
```bash
jupyter notebook notebooks/23001518_TranMinhDuc_Lab3_NLP.ipynb
```

### 3. Chạy từng cell theo thứ tự:
1. Import libraries và load model
2. Chọn 5,000 từ
3. Giảm chiều với PCA
4. Vẽ biểu đồ PCA
5. Giảm chiều với t-SNE
6. Vẽ biểu đồ t-SNE
7. So sánh PCA vs t-SNE
8. Trực quan hóa nhóm từ
9. Tìm và trực quan hóa từ gần nghĩa

---

## Kết luận

### Tổng kết kiến thức:

**1. Word Embeddings:**
- Biểu diễn từ dưới dạng **dense vectors** thay vì one-hot encoding
- Bắt được **mối quan hệ ngữ nghĩa** (từ gần nghĩa có vector gần nhau)
- Mô hình GloVe học từ **co-occurrence statistics** trên corpus khổng lồ
- Là **nền tảng** cho hầu hết các mô hình NLP hiện đại

**2. Kỹ thuật giảm chiều:**

| Phương pháp | PCA | t-SNE |
|-------------|-----|-------|
| **Loại** | Tuyến tính | Phi tuyến |
| **Tốc độ** | Nhanh O(n²d) | Chậm O(n² log n) |
| **Bảo toàn** | Toàn cục | Cục bộ |
| **Cluster** | Mờ | Rõ ràng |
| **Ổn định** | Deterministic | Stochastic |
| **Use case** | Preprocessing | Visualization |

**3. Quan sát về mối quan hệ ngữ nghĩa:**
- Các từ **cùng chủ đề tập trung gần nhau**: quốc gia, động vật, màu sắc, số
- Mô hình đã học được **semantic similarity** từ corpus

---
