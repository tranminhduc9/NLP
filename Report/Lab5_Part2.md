# Lab 5 Part 2: RNNs for Text Classification

## Mục tiêu
- Áp dụng **Recurrent Neural Networks (RNNs)** cho bài toán phân loại văn bản
- So sánh hiệu quả của các phương pháp biểu diễn văn bản: **TF-IDF**, **Word2Vec**, **Embedding**
- Xây dựng và đánh giá pipeline phân loại intent từ câu hỏi người dùng
- So sánh mô hình **LSTM** với embedding pre-trained và embedding học từ đầu
- Thực hành với **Keras/TensorFlow** để xây dựng deep learning models

---

## Dataset: HWU64

**Mô tả:**
- **HWU64 Intent Classification Dataset** - Bộ dữ liệu phân loại ý định từ câu hỏi người dùng
- **64 lớp intent** khác nhau (alarm_set, weather_query, music_play, reminder_create, ...)

**Phân chia dữ liệu:**
```
Train:      8,954 samples
Validation: 1,076 samples
Test:       1,076 samples
```

## Nhiệm vụ

### Chuẩn bị dữ liệu

**1. Load dataset:**

**2. Encode labels:**

**Lý do encode:**
- Chuyển text labels thành số (0-63) để phù hợp với neural networks
---

### Task 1: Pipeline TF-IDF + Logistic Regression

**Mô hình baseline đơn giản:**

```
Input Text
    ↓
TF-IDF Vectorizer (5000 features)
    ↓
Logistic Regression
    ↓
Predicted Intent
```

**Đặc điểm:**
- **TF-IDF**: Biểu diễn văn bản bằng TF-IDF
- **Logistic Regression**: Mô hình tuyến tính đơn giản
- **Ưu điểm**: Nhanh, dễ triển khai, baseline tốt
- **Nhược điểm**: Không bắt được thứ tự từ, không học được semantic meaning

**Kết quả:**
- **F1-score (Macro)**: 0.8353 (83.53%)
- Kết quả rất tốt cho một mô hình baseline đơn giản

---

### Task 2: Pipeline Word2Vec (Trung bình) + Dense Layer

**Cải tiến với Word Embeddings:**

#### Bước 1: Huấn luyện Word2Vec

**Tham số:**
- **vector_size=200**: Mỗi từ được biểu diễn bằng vector 200 chiều
- **window=5**: Xét 5 từ trước/sau để học context
- **sg=1**: Skip-gram model (dự đoán context từ target word)
- **epochs=20**: Huấn luyện 20 lần trên toàn bộ corpus

---

#### Bước 2: Chuyển câu thành vector trung bình

#### Bước 3: Xây dựng Neural Network

**Kiến trúc:**
```
Input (200 dims)
    ↓
Dense(256) + ReLU
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
Dense(128) + ReLU
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
Dense(64) + Softmax
    ↓
Output (64 classes)
```

**Các kỹ thuật:**
- **BatchNormalization**: Chuẩn hóa activation, tăng tốc độ học
- **Dropout(0.3)**: Regularization, tránh overfitting
- **ReLU**: Hàm kích hoạt phi tuyến
- **Softmax**: Chuyển output thành xác suất

---

#### Bước 4: Huấn luyện và đánh giá

**Kết quả:**
- **Test Accuracy**: ~75%
- **F1-score (Macro)**: 0.7751 (77.51%)
- **Test Loss**: 0.8287

**Phân tích:**
- Kết quả **thấp hơn** TF-IDF + Logistic Regression (83.53%)
- **Nguyên nhân**: Vector trung bình **mất thông tin thứ tự** từ trong câu

---

### Task 3: Mô hình Nâng cao (Embedding Pre-trained + LSTM)

**Cải tiến với RNN để bắt thứ tự từ:**

#### Bước 1: Tiền xử lý cho mô hình chuỗi

```python
# Tokenizer: Tạo vocabulary và chuyển text → sequences
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(df_train['text'])

train_sequences = tokenizer.texts_to_sequences(df_train['text'])
val_sequences = tokenizer.texts_to_sequences(df_val['text'])
test_sequences = tokenizer.texts_to_sequences(df_test['text'])

# Padding: Đảm bảo các chuỗi có cùng độ dài
max_len = 50
X_train_pad = pad_sequences(train_sequences, maxlen=max_len, padding='post')
X_val_pad = pad_sequences(val_sequences, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(test_sequences, maxlen=max_len, padding='post')
```


#### Bước 2: Tạo Embedding Matrix từ Word2Vec

**Mục đích:**
- Sử dụng **pre-trained embeddings** từ Word2Vec đã huấn luyện
- Mỗi word index → vector 200 chiều
- Từ không có trong Word2Vec → vector zeros

---

#### Bước 3: Xây dựng mô hình Bidirectional LSTM


**Kiến trúc:**
```
Input (50 tokens)
    ↓
Embedding Layer (vocab_size → 200) [Pre-trained, Frozen]
    ↓
SpatialDropout1D(0.2)
    ↓
Bidirectional LSTM(128) + return_sequences=True
    ↓
Bidirectional LSTM(64)
    ↓
Dense(128) + ReLU
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
Dense(64) + Softmax
```

**Các thành phần:**

1. **Embedding Layer:**

2. **SpatialDropout1D:**
   - Dropout toàn bộ feature maps
   - Hiệu quả hơn cho dữ liệu chuỗi

3. **Bidirectional LSTM:**
   - Đọc chuỗi **cả 2 chiều**: forward và backward
   - Bắt được context từ cả trước và sau
   - `return_sequences=True`: Trả về output cho mỗi bước

4. **Dropout & Recurrent Dropout:**
   - `dropout=0.2`: Hiệu chỉnh cho input
   - `recurrent_dropout=0.2`: Hiệu chỉnh cho recurrent connections

---

#### Bước 4: Huấn luyện với Callbacks

**Callbacks:**
- **EarlyStopping**: Tránh overfitting, tiết kiệm thời gian
- **ReduceLROnPlateau**: Điều chỉnh tốc độ học, cải thiện việc hội tụ

**Kết quả:**
- **Test Accuracy**: ~80%
- **F1-score (Macro)**: 0.8021 (80.21%)
- **Test Loss**: 0.6617

**Phân tích:**
- Cải thiện so với Word2Vec + Dense (77.51%) nhờ LSTM bắt được thứ tự từ
- Nhưng vẫn **thấp hơn TF-IDF baseline** (83.53%)
- **Nguyên nhân**: Sử dụng Embedding bằng Pre-trained Model nhưng chưa tối ưu với dữ liệu riêng này  

---

### Task 4: Mô hình Nâng cao (Embedding học từ đầu + LSTM)

**Học embeddings từ đầu thay vì dùng pre-trained:**

```python
lstm_model_scratch = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=200,              # Chiều Embedding
        input_length=max_len
    ),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='softmax')
])
```

**Khác biệt với Task 3:**
- **Task 3**: Embedding pre-trained từ Word2Vec, `trainable=False`
- **Task 4**: Embedding random initialization, `trainable=True`

---

#### Huấn luyện và đánh giá
**Kết quả:**
- **Test Accuracy**: ~84%
- **F1-score (Macro)**: 0.8428 (84.28%)
- **Test Loss**: 0.8287

**Phân tích:**
- **Tốt nhất** trong 4 mô hình
- Vượt qua TF-IDF baseline (83.53%) và Pre-trained + LSTM (83.31%)
- Vượt xa Word2Vec + Dense (77.51%) 
- **Lý do**: Embeddings được học **trực tiếp cho task phân loại intent** + LSTM 2 chiều giúp nắm rõ thứ tự từ

---

### Task 5: Đánh giá và So sánh

#### 5.1. Đánh giá định lượng

**Bảng kết quả F1-score và Loss:**

```

================================================================================
BẢNG TỔNG HỢP KẾT QUẢ F1-SCORE (MACRO) VÀ LOSS TRÊN TẬP KIỂM TRA
================================================================================
                      Pipeline  F1-score (Macro)  Test Loss
  TF-IDF + Logistic Regression          0.835298        NaN
        Word2Vec (Avg) + Dense          0.775115   0.616172
Embedding (Pre-trained) + LSTM          0.833110   0.616172
    Embedding (Scratch) + LSTM          0.842764   0.828690
================================================================================
```

**Xếp hạng:**
1. **Embedding (Scratch) + Bi-LSTM**: 84.28% F1-score
2. **TF-IDF + Logistic Regression**: 83.53% F1-score
3. **Embedding (Pre-trained) + Bi-LSTM**: 83.31% F1-score
4. **Word2Vec (Avg) + Dense**: 77.51% F1-score

---

#### 5.2. Đánh giá định tính

**Test cases:**

```python
test = [
    "set an alarm for 7 am tomorrow",
    "turn the volume down a little",
    "mute the audio completely",
    "what meetings do I have next Wednesday",
    "add a doctor's appointment at 4 pm",
    "remove my lunch event on Friday",
    "how do I cook fried rice",
    "convert 2 pm PST to CET",
    "what day is December 12th",
    "show me my unread emails",
    "send an email to Michael saying I'm on my way",
    "tell me a funny joke",
    "dim the living room lights to 20 percent",
    "turn on the hue lights in the kitchen",
    "what’s on my grocery list",
    "play some relaxing jazz music",
    "define the word perspective",
    "how much is 100 dollars in euros",
    "what’s the latest news today",
    "will it rain in London tomorrow"
]


test_label = [
    "alarm_set",
    "audio_volume_down",
    "audio_volume_mute",
    "calendar_query",
    "calendar_set",
    "calendar_remove",
    "cooking_recipe",
    "datetime_convert",
    "datetime_query",
    "email_query",
    "email_sendemail",
    "general_joke",
    "iot_hue_lightdim",
    "iot_hue_lighton",
    "lists_query",
    "play_music",
    "qa_definition",
    "qa_currency",
    "news_query",
    "weather_query"
]

```

**Kết quả:**
```
Task 1 (TF-IDF + LR):              19/20 đúng = 95.0%
Task 2 (Word2Vec + Dense):         18/20 đúng = 90.0%
Task 3 (Pre-trained + Bi-LSTM):    19/20 đúng = 95.0%
Task 4 (Scratch + Bi-LSTM):        20/20 đúng = 100.0%
```

**Phân tích:**
- Task 4 tốt nhất với 20/20 tests
- Task 1,3 chỉ sai 1 câu
- Task 2 thì thấp nhất

---

## Phân tích chi tiết

### 1. Tại sao TF-IDF + LR tốt hơn Word2Vec + Dense?

**Lấy vector trung bình gây mất thông tin:**
- Việc lấy vector trung bình không còn giữ được tầm quan trọng của từ.
- **TF-IDF** giữ được thông tin về tầm quan trọng của từ (IDF).
- **Logistic Regression** với TF-IDF features học được pattern tốt

### 2. Tại sao Embedding Scratch tốt hơn Pre-trained?

**Tối ưu hóa task:**
- **Pre-trained Word2Vec**: Học từ corpus tổng quát (Wikipedia, news, ...)
  - Ví dụ: "book" → embedding gần "novel", "author", "read"
  
- **Embedding từ Scratch**: Học từ HWU64 intent dataset
  - "book" trong context "book a flight" → embedding gần "reserve", "flight", "hotel"

### 3. Ý nghĩa của Bidirectional LSTM

**Bidirectional:**
```
Text: "book a flight to london"

Forward LSTM:  book → a → flight → to → london
               [h1] → [h2] → [h3] → [h4] → [h5]

Backward LSTM: london ← to ← flight ← a ← book
               [h5'] ← [h4'] ← [h3'] ← [h2'] ← [h1']

Output: concat([h5, h5']) → Context từ cả 2 chiều
```

### 4. Hiệu quả của Regularization

**Các kỹ thuật sử dụng:**

1. **Dropout (0.2-0.3)** và **Recurrent Dropout (0.2)**:
   - Tránh overfitting

2. **BatchNormalization**:
   - Tăng tốc training, học ổn định

5. **EarlyStopping**:
   - Dừng khi validation không cải thiện
   - Tránh overfitting

6. **ReduceLROnPlateau**:
   - Cải thiện độ hội tụ

---

## Kết luận

### Kết quả đạt được:
- Xây dựng và so sánh **4 pipeline** phân loại intent với độ chính xác 78-84%
- Hiểu rõ ưu/nhược điểm của **TF-IDF**, **Word2Vec**, **Pre-trained Embeddings**, **Learnable Embeddings**
- Thực hành với **Bidirectional LSTM** và các kỹ thuật regularization
- Đánh giá mô hình bằng **F1-score macro** và test cases thực tế

### Bài học:

#### 1. Về biểu diễn văn bản:
- **TF-IDF**: Đơn giản, nhanh, baseline tốt cho text classification
- **Word2Vec Averaging**: Mất thông tin thứ tự, kém hiệu quả
- **Embeddings + RNN**: Bắt được thứ tự từ, hiệu quả với dữ liệu dạng chuỗi

#### 2. Về Deep Learning:
- **Pre-trained embeddings** tốt khi **ít dữ liệu** hoặc **tasks giống training corpus**
- **Embeddings from scratch** tốt hơn khi có **đủ dữ liệu** và **task chuyên biệt**
- **Bidirectional LSTM** hiệu quả cho text classification
- **Regularization** quan trọng: Dropout, BatchNorm, EarlyStopping

