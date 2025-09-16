# Lab 2: Count Vectorization

## Mục tiêu
- Mục tiêu của Lab này là biểu diễn các văn bản dưới dạng **vector số**.  
- Thực hiện mô hình **Bag-of-Words** sử dụng `CountVectorizer`.  
- Lab này sẽ **tái sử dụng Tokenizer** từ Lab 1 để chuẩn hóa văn bản trước khi vector hóa.

## Cấu trúc dự án

NLP
│ .gitignore
│ README.md
│ requirements.txt
│
├───data
│ └───UD_English-EWT
│
├───Report
│       Lab1.md
│       Lab2.md
│
├───src
│ ├───core
│ │     dataset_loaders.py
│ │     interfaces.py
│ │     init.py
│ │
│ ├───preprocessing
│ │     regex_tokenizer.py
│ │     simple_tokenizer.py
│ │     init.py
│ │
│ └───representations
│       count_vectorizer.py
│       init.py
│
└───test
        lab1_test.py
        lab2_test.py
        init.py
## Nhiệm vụ

### Task 1: Vectorizer Interface
- Định nghĩa **abstract base class** `Vectorizer` trong `src/core/interfaces.py` với các phương thức:
  1. `fit(self, corpus: list[str])` – học **vocabulary** từ tập văn bản.
  2. `transform(self, documents: list[str]) -> list[list[int]]` – chuyển văn bản thành **vector đếm** dựa trên vocabulary.
  3. `fit_transform(self, corpus: list[str]) -> list[list[int]]` – thực hiện `fit` rồi `transform` trên cùng dữ liệu.

---

### Task 2: CountVectorizer Implementation
- Tạo lớp `CountVectorizer` trong `src/representations/count_vectorizer.py`.

#### Phương thức `fit`
- Khởi tạo một **set** để lưu các token duy nhất.  
- RegexTokenize từng **document** trong corpus và thêm các token vào set.  
- Sau khi xử lý xong tất cả document, tạo **dictionary `vocabulary_`** ánh xạ mỗi token với một **index duy nhất**.

---

#### Phương thức `transform`
- Với mỗi document:
  1. Tạo một **vector zeros** có độ dài bằng **số lượng token trong vocabulary**.  
  2. RegexTokenize document.  
  3. Với mỗi token:
     - Nếu token tồn tại trong `vocabulary_`, **tăng count tại index tương ứng**.  
- Trả về **danh sách các vector** cho toàn bộ corpus.

## Cách chạy:
1. Cài đặt môi trường:
   ```bash
   pip install -r requirements.txt
    ```
2. Chạy test:
    - Lab 2:
    ```bash
    python -m test.lab2_test
    ```

## Kết quả
    ```python
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    ```

### Kết quả
[[1, 0, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]]


### Phân tích
 - Tokenizer sử dụng là RegexTokenizer.
 - Mỗi vector tương ứng với số lần xuất hiện của token trong vocabulary học được từ corpus.
 - Đây là biểu diễn Bag-of-Words cho mỗi document.

