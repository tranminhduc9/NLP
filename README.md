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

## 📌 Lab 3: Visualize model with PCA and T-SNE
### Nội dung thực hiện



## 📌 Lab 4: Word Embedding
### Nội dung thực hiện
  1. lab4_test.py
  - Xây dựng lớp WordEmbedder để làm việc với word embeddings.
  - Cài mô hình pre-trained GloVe (glove-wiki-gigaword-50) bằng thư viện gensim.
  - Cài đặt các hàm xử lý:
      + get_vector(word): Lấy vector của một từ.
      + get_similarity(word1, word2): Tính độ tương đồng cosine giữa hai từ.
      + get_most_similar(word): Tìm các từ gần nghĩa nhất.
  - Hàm embed_document(document): Biểu diễn vector của một văn bản bằng cách lấy trung bình cộng các vector của các từ trong văn bản đó.
  - Tạo test (`test/lab4_test.py`) để chạy với ví dụ.

  2. lab4_embedding_trainning_demo.py
  - Xây dựng lớp StreamSentences để đọc dữ liệu lớn từ file theo từng đoạn (Tiết kiệm RAM), tự động tách câu/đoạn và tokenize.
  - Sử dụng thư viện gensim để huấn luyện mô hình Word2Vec từ trên corpus UD_English-EWT.
  - Tạo `test/lab4_embedding_trainning_demo.py` để chạy demo:
    + Tìm các từ có độ tương đồng cao (most_similar).
    + Giải bài toán quan hệ từ (Anology) (ví dụ: king - man + queen = ?).

  3. lab4_spark_word2vec_demo.py
  - Sử dụng Apache Spark và thư viện MLlib để xử lý dữ liệu lớn.
  - Khởi tạo SparkSession và làm việc với tập dữ liệu lớn.
  - Đọc và tiền xử lý dữ liệu văn bản bằng các phép biến đổi của Spark DataFrame:
    + Chuyển chữ thường (lower).
    + Loại bỏ ký tự đặc biệt (regexp_replace).
  - Sử dụng Tokenizer của Spark ML để tách từ.
  - Huấn luyện mô hình Word2Vec trên DataFrame đã xử lý bằng pyspark.ml.feature.Word2Vec.
  - Sử dụng phương thức findSynonyms của mô hình đã huấn luyện để tìm các từ tương đồng.
  - Tạo `test/lab4_spark_word2vec_demo.py` để thực thi toàn bộ pipeline xử lý dữ liệu và huấn luyện mô hình trên Spark.

### Học được
  1. lab4_test.py
    - Sử dụng các mô hình word embedding pre-trained như GloVe để biểu diễn vector của từ.
    - Tìm từ tương tự, tính độ tương đồng.
    - Tạo vector cho văn bản (document embedding) từ các word embedding có sẵn.
  2. lab4_embedding_trainning_demo.py
    - Cách huấn luyện một mô hình Word2Vec từ đầu bằng thư viện gensim.
    - Sự khác biệt giữa việc sử dụng mô hình pre-trained (`lab4_test.py`) và tự huấn luyện mô hình word embedding trên dữ liệu riêng.
    - Kỹ thuật xử lý dữ liệu lớn (corpus) để huấn luyện mô hình.
  3. lab4_spark_word2vec_demo.py
    - Cách xây dựng một pipeline xử lý ngôn ngữ tự nhiên với Apache Spark.
    - Sự khác biệt về tốc độ làm việc với gensim so với Spark MLlib trên dữ liệu quy mô lớn.
    - Cách sử dụng các thư viện của Spark ML như Tokenizer và Word2Vec.

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

    - Lab 3: Chạy từng cell code trong `23001518_TranMinhDuc_Lab3_NLP.ipynb` trong notebooks

    - Lab 4\ Lab_4_ebeeding_trainning \ Lab4_spark_word2vec:
    ```bash
    python -m test.lab4_test
    ```
    ```bash
    python -m test.lab4_embedding_trainning_demo
    ```
    ```bash
    python -m test.lab4_spark_word2vec_demo
    ```


