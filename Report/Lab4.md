# Lab 4: Word Embeddings

## Mục tiêu
- Mục tiêu của Lab này là hiểu và làm việc với **Word Embeddings** - biểu diễn vector của từ trong không gian nhiều chiều.
- Sử dụng mô hình **pre-trained GloVe** để lấy vector biểu diễn của từ.
- Tự **huấn luyện mô hình Word2Vec** từ corpus.
- Sử dụng **Apache Spark MLlib** để xử lý và huấn luyện mô hình trên dữ liệu lớn.
- Tìm hiểu các tác vụ: tính **độ tương đồng**, tìm **từ gần nghĩa**, giải **bài toán quan hệ từ** (word analogy).

## Cấu trúc dự án

```
NLP
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│   │   c4-train.00000-of-01024-30K.json
│   │
│   └───UD_English-EWT
│           en_ewt-ud-dev.conllu
│           en_ewt-ud-dev.txt
│           en_ewt-ud-test.conllu
│           en_ewt-ud-test.txt
│           en_ewt-ud-train.conllu
│           en_ewt-ud-train.txt
│           LICENSE.txt
│           README.md
│
├───Report
│       Lab1.md
│       Lab2.md
│       Lab4.md
│
├───results
│       word2vec_ewt.model
│
├───src
│   ├───core
│   │      dataset_loaders.py
│   │      interfaces.py
│   │      __init__.py
│   │
│   ├───preprocessing
│   │      regex_tokenizer.py
│   │      simple_tokenizer.py
│   │      __init__.py
│   │
│   └───representations
│          count_vectorizer.py
│          word_embedder.py
│          __init__.py
│
└───test
        lab1_test.py
        lab2_test.py
        lab4_test.py
        lab4_embedding_trainning_demo.py
        lab4_spark_word2vec_demo.py
        __init__.py
```

## Nhiệm vụ

### Task 1: WordEmbedder với Pre-trained Model
- Tạo lớp `WordEmbedder` trong `src/representations/word_embedder.py`.
- Sử dụng thư viện **gensim** để tải mô hình pre-trained **GloVe** (`glove-wiki-gigaword-50`).

#### Các phương thức chính:
1. **`get_vector(word: str)`** – Lấy vector biểu diễn của một từ.
   - Trả về vector nếu từ có trong từ điển.
   - Trả về `None` nếu từ không tồn tại.

2. **`get_similarity(word1: str, word2: str)`** – Tính độ tương đồng giữa hai từ.
   - Sử dụng **cosine similarity**.
   - Công thức: `cosine(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)`

3. **`get_most_similar(word: str, topn: int = 10)`** – Tìm các từ gần nghĩa nhất.
   - Sử dụng phương thức `most_similar` của gensim.
   - Trả về danh sách `topn` từ có vector gần nhất.

4. **`embed_document(document: str)`** – Biểu diễn vector của một văn bản.
   - Sử dụng **RegexTokenizer** từ Lab 1 để tách từ.
   - Tính **vector trung bình** của tất cả các từ trong văn bản.
   - Trả về vector 0 nếu không có từ nào trong từ điển.

#### Document Embedding:
- Vector văn bản = **Trung bình vector các từ**.
- Đơn giản nhưng mất thứ tự từ.
- Các phương pháp tốt hơn: Doc2Vec, BERT embeddings.
---

### Task 2: Huấn luyện mô hình Word2Vec với Gensim
- Tạo script `lab4_embedding_trainning_demo.py` trong thư mục `test/`.

#### Xử lý dữ liệu lớn với StreamSentences:
- Tạo lớp `StreamSentences` để đọc dữ liệu theo từng đoạn (tiết kiệm RAM).
- Đọc file `en_ewt-ud-train.txt` từ corpus **UD_English-EWT**.
- Sử dụng **RegexTokenizer** để tokenize từng đoạn văn bản.

#### Huấn luyện Word2Vec:
- Sử dụng thư viện **gensim.models.Word2Vec**.
- Các tham số quan trọng:
  - `vector_size=100`: Số chiều của vector từ.
  - `window=5`: Kích thước cửa sổ ngữ cảnh.
  - `min_count=3`: Số lần xuất hiện tối thiểu của từ.
  - `workers=4`: Số luồng xử lý song song.
  - `sg=1`: Skip-gram (1) hoặc CBOW (0).

#### Demo mô hình:
1. **Tìm từ gần nghĩa** với từ `"computer"`.
2. **Giải bài toán quan hệ từ** (word analogy):
   - `"king" - "man" + "queen" = ?`

#### Lưu mô hình:
- Lưu mô hình đã huấn luyện vào `results/word2vec_ewt.model`.

---

### Task 3: Huấn luyện Word2Vec với Apache Spark
- Tạo script `lab4_spark_word2vec_demo.py` trong thư mục `test/`.

#### Khởi tạo Spark:
- Sử dụng **SparkSession** với cấu hình bộ nhớ driver 4GB.

#### Đọc và tiền xử lý dữ liệu:
- Đọc file JSON `c4-train.00000-of-01024-30K.json`.
- Tiền xử lý văn bản:
  1. Chuyển chữ thường (`lower`).
  2. Loại bỏ ký tự đặc biệt (`regexp_replace`).
- Sử dụng **Tokenizer** của Spark ML để tách từ.

#### Huấn luyện Word2Vec:
- Sử dụng **pyspark.ml.feature.Word2Vec**.
- Các tham số:
  - `vectorSize=100`: Kích thước vector.
  - `minCount=5`: Số lần xuất hiện tối thiểu.

#### Tìm từ tương đồng:
- Sử dụng phương thức `findSynonyms` để tìm 5 từ gần nghĩa nhất với `"computer"`.

---

## Cách chạy

### 1. Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

### 2. Chạy test:

#### Lab 4 - WordEmbedder với Pre-trained Model:
```bash
python -m test.lab4_test
```

#### Lab 4 - Huấn luyện Word2Vec với Gensim:
```bash
python -m test.lab4_embedding_trainning_demo
```

#### Lab 4 - Huấn luyện Word2Vec với Spark:
```bash
python -m test.lab4_spark_word2vec_demo
```

**Lưu ý**: Spark yêu cầu Java JDK 8+ và biến môi trường `JAVA_HOME` được cấu hình đúng.

---

## Kết quả

### 1. Lab4_test.py - WordEmbedder với GloVe

#### Input:
```python
word1 = "king"
word2 = "queen"
word3 = "man"
document = "The queen rules the country."
```

#### Kết quả:

**1. Vector của từ 'king':**
```
[ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
  0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
 -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
 -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
  0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
 -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
 -0.51042 ]
```

**2. Độ tương đồng:**
```
Độ tương đồng giữa 'king' và 'man': 0.530937671661377
Độ tương đồng giữa 'king' và 'queen': 0.7839041948318481
```

**3. Từ tương tự với 'computer':**
```
[('computers', 0.9165045022964478), ('software', 0.8814994096755981), ('technology', 0.8525559306144714), ('electronic', 0.812586784362793), ('internet', 0.8060454726219177), ('computing', 0.802603542804718), ('devices', 0.8016185760498047), ('digital', 0.7991792559623718), ('applications', 0.7912740707397461), ('pc', 0.7883161306381226)]
```

**4. Vector biểu diễn tài liệu:**
```
[ 0.04564168  0.36530998 -0.55974334  0.04014383  0.09655549  0.15623933
 -0.33622834 -0.12495166 -0.01031508 -0.5006717   0.18690467  0.17482166
 -0.268985   -0.03096624  0.36686516  0.29983264  0.01397333 -0.06872118
 -0.3260683  -0.210115    0.16835399 -0.03151734 -0.06204716  0.04301083
 -0.06958768 -1.7792168  -0.54365396 -0.06104483 -0.17618     0.009181
  3.3916333   0.08742473 -0.4675417  -0.213435    0.02391887 -0.04470453
  0.20636833 -0.12902866 -0.28527132 -0.2431805  -0.3114423  -0.03833717
  0.11977985 -0.01418401 -0.37086335  0.22069354 -0.28848937 -0.36188802
 -0.00549529 -0.46997246]
```

---

### 2. Lab4_embedding_trainning_demo.py - Huấn luyện Word2Vec

#### Kết quả:

**Training Word2Vec model...**

...
Mô hình lưu tại results/word2vec_ewt.model
```

**Tìm từ gần nghĩa với 'computer':**
```
  - impress         (Similarity: 0.9863)
  - easiest         (Similarity: 0.9845)
  - registered      (Similarity: 0.9843)
  - goals           (Similarity: 0.9841)
  - writing         (Similarity: 0.9840)
  - crop            (Similarity: 0.9840)
  - frequent        (Similarity: 0.9840)
  - visiting        (Similarity: 0.9836)
  - wheel           (Similarity: 0.9836)
  - memory          (Similarity: 0.9834)
```

**'king' -> 'man' thì 'queen' -> ?**
```
  -> Đáp án: 'manager'
```

---

### 3. Lab4_spark_word2vec_demo.py - Spark MLlib

#### Kết quả:

**5 từ tương tự nhất với 'computer':**
```
+---------+------------------+
|     word|        similarity|
+---------+------------------+
|  desktop|0.7540328502655029|
|computers|0.6907596588134766|
|       pc|0.6904626488685608|
|  uwowned|0.6829250454902649|
|   laptop| 0.665536105632782|
+---------+------------------+
```

---

## Phân tích

### 1. Pre-trained Model (GloVe):
- **Ưu điểm**: Sử dụng nhanh, không cần huấn luyện, vector chất lượng cao từ corpus lớn (Wikipedia).
- **Nhược điểm**: Không tùy chỉnh cho domain cụ thể, vocabulary cố định.
- **Phù hợp**: Các ứng dụng cần triển khai nhanh, dữ liệu domain chung.

### 2. Tự huấn luyện Word2Vec (Gensim):
- **Ưu điểm**: Tùy chỉnh cho corpus riêng, học được từ vựng domain-specific.
- **Nhược điểm**: Cần dữ liệu lớn, thời gian huấn luyện lâu hơn, kết quả đôi khi không như kỳ vọng.
- **Phù hợp**: Khi có corpus domain cụ thể.

### 3. Spark MLlib Word2Vec:
- **Ưu điểm**: Xử lý được dữ liệu rất lớn, tận dụng xử lý phân tán.
- **Nhược điểm**: Cấu hình phức tạp hơn, overhead khi dữ liệu nhỏ.
- **Phù hợp**: Big data, production environment với cluster Spark.

---

## Kết luận

- **Word Embeddings** là bước tiến lớn so với Bag-of-Words: bắt được **ngữ nghĩa** và **quan hệ từ**.
- **Pre-trained models** (GloVe, Word2Vec, FastText) hiệu quả cho hầu hết bài toán NLP.
- **Tự huấn luyện** khi cần tùy chỉnh cho domain cụ thể.
- **Spark MLlib** là lựa chọn tốt khi làm việc với **Big Data**.
