# Lab 5: Text Classification

## Mục tiêu
- Mục tiêu của Lab này là xây dựng một pipeline phân loại văn bản hoàn chỉnh, từ văn bản thô đến một mô hình học máy đã được huấn luyện, sử dụng các kỹ thuật tách từ (tokenization) và biểu diễn đặc trưng (vectorization) đã học trong các lab trước.

- Tự xây dựng hệ thống phân loại văn bản dựa trên mô hình học máy Logistic Regression
- Triển khai **Pipeline Machine Learning** bằng Pyspark.
- So sánh kết quả của các mô hình khác nhau: **Logistic Regression**, **GBTClassifier**, **MultilayerPerceptronClassifier**.
- Tối ưu hóa pipeline để cải thiện độ chính xác của mô hình.

## Cấu trúc dự án

```
NLP
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│       sentiments.csv
│
├───Report
│       Lab5.md
│
├───src
│   ├───core
│   │      dataset_loaders.py
│   │      interfaces.py
│   │      __init__.py
│   │
│   ├───preprocessing
│   │      regex_tokenizer.py
│   │      __init__.py
│   │      
│   │
│   └───representations
│         tfidf_vectorizer.py
│          __init__.py 
│          
│
└───test
        lab5_test.py
        lab5_spark_sentiment_analysis.py
        lab5_improvement_test.py
        __init__.py
```

## Cách chạy

### 1. Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

**Lưu ý**: Spark yêu cầu:
- Java JDK 8+ (đặt biến môi trường `JAVA_HOME`).
- PySpark đã được cài đặt.

### 2. Chạy test:
#### Lab 5 - TextClassifier với Scikit-learn:
```bash
python -m test.lab5_test
```

#### Lab 5 - Baseline PySpark Sentiment Analysis:
```bash
python -m test.lab5_spark_sentiment_analysis
```

#### Lab 5 - So sánh các mô hình cải tiến:
```bash
python -m test.lab5_improvement_test
```

## Nhiệm vụ

### Task 1: Data Preparation

- Tạo một bộ dữ liệu nhỏ gồm 6 mẫu và nhãn tương ứng như sau:

```python
    texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
    ]

    labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative
```

- Sử dụng TF-IDF Vectorizer (ở Lab trước) để vector hóa dữ liệu văn bản thành dữ liệu số.

### Task 2: TextClassifier với Scikit-learn
- Tạo lớp `TextClassifier` trong `src/models/text_classifier.py`.
- Sử dụng **Logistic Regression** từ scikit-learn làm mô hình cơ bản.
- Tích hợp với **TfidfVectorizer** đã xây dựng.

#### Các phương thức chính:
1. **`fit(texts: List[str], labels: List[int])`**: Huấn luyện mô hình.
   - Chuyển văn bản thành vector TF-IDF.
   - Huấn luyện Logistic Regression trên dữ liệu đã vector hóa.

2. **`predict(texts: List[str])`**: Dự đoán nhãn cho văn bản mới.
   - Chuyển văn bản thành vector TF-IDF.
   - Sử dụng mô hình đã huấn luyện để dự đoán.

3. **`evaluate(y_true: List[int], y_pred: List[int])`** – Đánh giá mô hình.
   - Tính các metrics: **accuracy**, **precision**, **recall**, **f1-score**.
   - Sử dụng các hàm từ `sklearn.metrics`.

#### Demo:
- Tạo script `lab5_test.py` để demo classifier.
- Sử dụng dữ liệu mẫu với 6 câu (positive/negative).
- Chia train/test với tỷ lệ 80/20.
- Hiển thị kết quả vector hóa, dự đoán và đánh giá.

### Task 3: Kết quả và đánh giá

#### Input:
```python
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
```

#### Kết quả Vector hóa:
```
Vector hóa văn bản:

Text: This movie is fantastic and I love it!
Vector: [0. 0. 0. ... 0. 0. 0.]

Text: I hate this film, it's terrible.
Vector: [0. 0. 0. ... 0. 0. 0.]

...
```

#### Train/Test Split:
```
Dữ liệu train: 
 ['What a waste of time, absolutely boring.', 'Highly recommend this, a masterpiece.', 'This movie is fantastic and I love it!', 'Could not finish watching, so bad.']
Dữ liệu test:
 ['The acting was superb, a truly great experience.', "I hate this film, it's terrible."]
```

#### Kết quả dự đoán và đánh giá:
```
Dự đoán nhãn [1 1]
Nhãn thật [1, 0]
Kết quả đánh giá:
 {'accuracy': 0.5, 'precision': 0.5, 'recall': 1.0, 'f1_score': 0.6666666666666666}
```


---

### Baseline Sentiment Analysis với PySpark
- Tạo script `lab5_spark_sentiment_analysis.py` trong thư mục `test/`.
- Xây dựng pipeline phân tích cảm xúc cơ bản với **PySpark MLlib**.

#### Khởi tạo Spark:
- Sử dụng **SparkSession** để khởi tạo môi trường Spark.

#### Đọc và tiền xử lý dữ liệu:
- Đọc file CSV `sentiments.csv` với cột `text` và `sentiment`.
- Chuyển đổi nhãn từ {-1, 1} thành {0, 1}:
- Loại bỏ các dòng có giá trị null trong cột `sentiment`.

#### Xây dựng Pipeline:
Pipeline gồm các bước:
1. **Tokenizer**: Tách văn bản thành từ.
2. **StopWordsRemover**: Loại bỏ stop words.
3. **HashingTF**: Chuyển từ thành vector TF (Term Frequency) với 10,000 features.
4. **IDF**: Tính trọng số IDF (Inverse Document Frequency).
5. **LogisticRegression**: Mô hình phân loại với `maxIter=10`, `regParam=0.001`.

#### Huấn luyện và đánh giá:
- Chia dữ liệu train/test với tỷ lệ 80/20.
- Huấn luyện pipeline trên dữ liệu train.
- Đánh giá trên test set với các metrics:
  - **Accuracy** (độ chính xác tổng thể)
  - **Precision** (độ chính xác của dự đoán positive)
  - **Recall** (khả năng tìm ra các mẫu positive)
  - **F1-score** (trung bình điều hòa của precision và recall)

#### Kết quả

```
Khởi tạo Spark session
Setting default log level to "WARN".

Tạo pipeline và huấn luyện mô hình...

Kết quả đánh giá mô hình:
{
    'accuracy': 0.7225085910652921,
    'precision': 0.7243196623920588,
    'recall': 0.7225085910652921,
    'f1_score': 0.7233489411941975
}
```

**Phân tích**:
- **Accuracy**: 72.25% - Mô hình baseline đạt kết quả khá tốt với pipeline đơn giản.
- **Precision**: 72.43% - Trong các mẫu được dự đoán là positive, 72.43% là chính xác.
- **Recall**: 72.25% - Mô hình tìm được 72.25% các mẫu positive thực tế.
- **F1-score**: 72.33% - Cân bằng tốt giữa precision và recall.

---

### Task 3: Cải tiến và So sánh Mô hình
- Tạo script `lab5_improvement_test.py` để so sánh nhiều mô hình.
- Mục tiêu: cải tiến mô hình baseline.

#### Tiền xử lý văn bản:
Tạo hàm `preprocess_text_dataframe()` với các bước:
1. Chuyển chữ thường (`lower`).
2. Loại bỏ URL: `http\S+|www\S+|https\S+`.
3. Loại bỏ HTML tags: `<.*?>`.
4. Loại bỏ ký tự đặc biệt, chỉ giữ chữ cái: `[^a-zA-Z\s]`.
5. Loại bỏ khoảng trắng thừa: `\s+`.
6. Trim khoảng trắng đầu/cuối.

#### Các Pipeline thử nghiệm:

**1. Baseline Pipeline:**
- Sử dụng pipeline từ Task 2.
- HashingTF (10,000 features) + IDF + Logistic Regression.

**2. Improved Pipeline (GBTClassifier):**
- Sử dụng **RegexTokenizer** để tokenize tốt hơn.
- HashingTF (2,000 features) + IDF.
- **GBTClassifier** (Gradient Boosted Trees):
  - `maxIter=100`: Số lượng cây.
  - Mô hình ensemble, thường cho kết quả tốt hơn Logistic Regression.

**3. Neural Networks Pipeline:**
- Sử dụng **MultilayerPerceptronClassifier** (MLP).
- HashingTF (**5,000 features**) + IDF.
- Kiến trúc mạng: `[5000, 64, 32, 2]`
  - Input layer: 5,000 features
  - Hidden layer 1: 64 neurons
  - Hidden layer 2: 32 neurons
  - Output layer: 2 classes (positive/negative)
- `maxIter=150`, `blockSize=128`, `seed=42`.

#### So sánh kết quả:
- Huấn luyện cả 3 mô hình trên cùng train/test split.
- Đánh giá với 4 metrics: accuracy, precision, recall, f1_score.
- Hiển thị bảng so sánh để xác định mô hình tốt nhất.


#### Kết quả:
```
Khởi tạo Spark session
Tạo Baseline_model...
Tạo Improved_model...
Tạo Neural_networks_model...

KẾT QUẢ SO SÁNH CÁC MÔ HÌNH
Metric          Baseline             Improved             Neural Network      
--------------------------------------------------------------------------------
accuracy        0.7225               0.7629               0.7646
precision       0.7243               0.7575               0.7739
recall          0.7225               0.7629               0.7646
f1_score        0.7233               0.7521               0.7676
```
---

**Phân tích chi tiết**:

1. **Baseline Model (HashingTF + Logistic Regression)**:
   - Accuracy: 72.25% - Kết quả chấp nhận được cho pipeline đơn giản.
   - Ưu điểm: Huấn luyện nhanh, dễ triển khai, phù hợp làm baseline.

2. **Improved Model (Tiền xử lý + HashingTF + GBTClassifier)**:
   - Accuracy: 76.29% - Cải thiện **+4.04%** so với baseline.
   - GBT mô hình phi tuyến bắt được quan hệ phức tạp giữa các đặc trưng..

3. **Neural Networks Model (Tiền xử lý + HashingTF + MLP)**:
   - Accuracy: 76.46% - **Tốt nhất** (+4.21% so với baseline).
   - Kiến trúc [5000, 64, 32, 2] với 150 iterations cho kết quả vượt trội.
   - Precision cao nhất (77.39%), phân biệt tốt giữa các lớp.
   - Trade-off: Thời gian huấn luyện lâu nhất.

---

**Phân tích hiệu quả**
1. Tiền xử lý văn bản - Loại bỏ nhiễu
- **Giảm nhiễu**: URL (`http://...`) và HTML tags (`<b>`, `<div>`) không mang thông tin cảm xúc, chỉ gây nhiễu.
- **Tăng tính nhất quán**: Chuẩn hóa giúp các từ giống nhau được nhận diện đồng nhất.

2. Điều chỉnh số lượng features (HashingTF)
**Quan sát từ thực nghiệm**:
- Baseline: 10,000 features
- Improved (GBT): 2,000 features  
- Neural Network: **5,000 features** → accuracy cao nhất

**Nhận xét**:
- **Quá nhiều features (10,000)**: 
  - Tăng độ phức tạp, dễ overfit với dataset nhỏ
  - Nhiều features không có thông tin (sparse)
  
- **Quá ít features (2,000)**:
  - Collision cao (nhiều từ khác nhau hash vào cùng bucket)
  - Mất thông tin quan trọng
  
- **Tối ưu (5,000)**:
  - Cân bằng giữa thông tin và độ phức tạp
  - Giảm collision, giữ được đủ từ vựng quan trọng

**Kết luận**: Số features cần điều chỉnh phù hợp với kích thước dataset.

3. Mô hình phức tạp hơn 

**GBTClassifier (Gradient Boosted Trees)**
- **Ensemble learning**: Kết hợp nhiều decision trees, mỗi cây học từ lỗi của cây trước.
- **Xử lý phi tuyến**: Bắt được pattern phức tạp mà Logistic Regression (tuyến tính) không thể.

**Neural Network (MLP)**
- **Học representation nhiều tầng**: 5000 -> 64 -> 32 -> 2 
- **Phi tuyến mạnh mẽ**: Cho phép học mối quan hệ phức tạp giữa features.

---

## Kết luận


### Kết quả đạt được:
- Hiểu được Pipeline phân loại văn bản: Văn bản đầu vào -> Tokenization -> Vectorization -> Xây dựng mô hình học máy -> Dự đoán
- Đánh giá mô hình thông qua các chỉ số: **accuracy** , **precision**, **recall** và **f1_score**
- Xây dựng Pipeline phân loại văn bản với PySpark.
- Cải thiện kết quả bài toán thông qua việc tiền xử lý và sử dụng các mô hình mạnh mẽ hơn.

### Học được
  1. **lab5_test.py**
    - Cách xây dựng pipeline phân loại văn bản hoàn chỉnh.
    - Tích hợp tokenizer, vectorizer vào classifier.
    - Sử dụng các metrics đánh giá mô hình phân loại.
  
  2. **lab5_spark_sentiment_analysis.py**
    - Xây dựng pipeline Machine Learning với PySpark MLlib.
    - Xử lý dữ liệu văn bản quy với Spark DataFrame.
  
  3. **lab5_improvement_test.py**
    - Tầm quan trọng của **tiền xử lý văn bản** và **số lượng features** (HashingTF) trong việc cải thiện accuracy.
    - So sánh hiệu suất các mô hình: Logistic Regression vs GBTClassifier vs Neural Network.
    - Trade-off giữa **độ chính xác** và **thời gian huấn luyện**.



