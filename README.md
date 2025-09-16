## Thông tin cá nhân
- **Họ và tên**: Trần Minh Đức
- **Mã sinh viên (MSV)**: 23001518
- **Lớp**: 68KHDL  
- **Ngày sinh**: 09/12/2005  

---

## 📌 Lab 1: Tokenization
### Nội dung thực hiện
- Xây dựng **interface `Tokenizer`**.
- Cài đặt **SimpleTokenizer**:
  - Chuyển chữ thường.
  - Tách từ theo khoảng trắng.
  - Xử lý tách dấu câu đơn giản (.,!?).
- Cài đặt **RegexTokenizer**:
  - Sử dụng regex `\w+|[^\w\s]` để tách từ và dấu câu.
- Tạo `Lab1/main.py` để chạy thử trên ví dụ và dataset `UD_English-EWT`.

###  Học được
- Sự khác biệt giữa **tokenizer thủ công** và **tokenizer regex**.


---

## 📌 Lab 2: Vectorization
###  Nội dung thực hiện
- Xây dựng **interface `Vectorizer`** với 3 phương thức:
  - `fit(corpus)`
  - `transform(documents)`
  - `fit_transform(corpus)`
- Cài đặt **CountVectorizer**:
  - Nhận vào một `Tokenizer`.
  - Học **vocabulary** từ corpus.
  - Biến đổi văn bản thành **bag-of-words vector**.
- Tạo test (`test/lab2_test.py`) để chạy với ví dụ.

### Học được
- Cách cài đặt thủ công mô hình Bag-of-Words.
- Cách tích hợp tokenizer vào vectorizer.

---

## ⚙️ Cách chạy
1. Cài đặt môi trường:
   ```bash
   pip install -r requirements.txt
    ```
2. Chạy test:
    - Lab 1:
    ```bash
    python -m test.lab1_test
    ```
    - Lab 2:
    ```bash
    python -m test.lab2_test  
    ```
