# Lab 5 Part 4: RNNs for Named Entity Recognition (NER)

## Mục tiêu
- Áp dụng **Recurrent Neural Networks (RNNs)** cho bài toán **Named Entity Recognition (NER)**
- Hiểu cách xử lý **sequence labeling** với PyTorch cho bài toán NER
- Xây dựng pipeline hoàn chỉnh: load data → preprocess → build model → train → evaluate
- Thực hành với **PyTorch Dataset**, **DataLoader**, **padding**, và **masking**
- Sử dụng **Bidirectional LSTM** để cải thiện độ chính xác
- Đánh giá với các metrics: Accuracy, Precision, Recall, F1-score

---

## Bài toán Named Entity Recognition (NER)

**NER (Named Entity Recognition):**
- Nhận dạng và phân loại các thực thể có tên trong văn bản
- Là bài toán **sequence labeling**: mỗi token được gán 1 nhãn thực thể

**Ví dụ:**
```
Input:  Barack Obama was born in Hawaii
Output: B-PER  I-PER O   O    O  B-LOC
```

**Các nhãn NER (CoNLL-2003):**
- **B-PER, I-PER**: Person (Người)
- **B-ORG, I-ORG**: Organization (Tổ chức)
- **B-LOC, I-LOC**: Location (Địa điểm)
- **B-MISC, I-MISC**: Miscellaneous (Khác)
- **O**: Outside (không phải thực thể)

**Lược đồ BIO:**
- **B** (Begin): Từ đầu tiên của thực thể
- **I** (Inside): Các từ tiếp theo trong thực thể
- **O** (Outside): Không thuộc thực thể nào

---

## Nhiệm vụ

### Task 1: Chuẩn bị dữ liệu

#### 1.1. Tải dữ liệu CoNLL-2003

**Dataset CoNLL-2003:**
- Dataset chuẩn cho bài toán NER
- Chứa các bài báo tiếng Anh từ Reuters
- 3 splits: train, validation, test

**Cấu trúc dữ liệu:**
- Train: 14,041 câu
- Validation: 3,250 câu  
- Test: 3,453 câu

**Định dạng dữ liệu:**
- `tokens`: Danh sách các từ trong câu
- `ner_tags`: Danh sách các ID nhãn tương ứng (0-8)

---

#### 1.2. Tiền xử lý dữ liệu

**Chuyển đổi nhãn số sang nhãn string:**

Đầu tiên, ta lấy danh sách ánh xạ từ ID sang nhãn string với 9 nhãn NER gồm: 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'. 
Sau đó, với mỗi câu trong dataset, ta chuyển đổi từng ID nhãn thành nhãn string tương ứng.

**Ví dụ câu đầu tiên:**
```
Tokens: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
Tags:   ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

---

#### 1.3. Xây dựng từ điển

**Tạo word_to_ix và tag_to_ix:**

- Khởi tạo word_to_ix với hai token đặc biệt là PAD (giá trị 0) và UNK (giá trị 1). Sau đó duyệt qua tất cả các câu trong tập huấn luyện, với mỗi từ chưa có trong từ điển thì thêm vào với index là độ dài hiện tại của từ điển. 
- Đối với tag_to_ix, tạo ánh xạ từ tên nhãn sang index dựa trên danh sách label_names, sau đó thêm token PAD với index cuối cùng.

**Kết quả:**
```
Kích thước word_to_ix: 23,624  (từ vựng)
Kích thước tag_to_ix: 10       (9 NER tags + PAD)
```

---

### Task 2: Tạo PyTorch Dataset và DataLoader

#### 2.1. NERDataset

**Custom Dataset class:**

Lớp `NERDataset` là một custom Dataset dùng cho bài toán NER trong PyTorch.  
Nó thực hiện các chức năng sau:

- Input:
  - `sentences`: Danh sách các câu (mỗi câu là list of tokens)
  - `tags`: Danh sách các nhãn NER tương ứng
  - `word_to_ix`, `tag_to_ix`: Từ điển chuyển đổi
  
- Chuyển từng câu thành hai tensor:
  - **word_indices**: dãy các chỉ số từ (word ID) dựa theo `word_to_ix`
  - **tag_indices**: dãy các chỉ số nhãn (tag ID) dựa theo `tag_to_ix`
  
- Tự động xử lý từ không có trong từ điển bằng token **`<UNK>`**

- Hỗ trợ PyTorch DataLoader bằng cách định nghĩa:
  - `__len__()` → trả về số lượng câu
  - `__getitem__()` → trả về (word_tensor, tag_tensor) của một câu

Lớp NERDataset được khởi tạo với các tham số sentences, tags, word_to_ix và tag_to_ix. Khi truy xuất một mẫu, hàm __getitem__ lấy câu và nhãn tương ứng theo index, sau đó chuyển đổi mỗi từ trong câu thành word index (nếu từ không có trong từ điển thì dùng UNK token), đồng thời chuyển mỗi nhãn thành tag index. Cuối cùng trả về hai tensor: word_indices và tag_indices với kiểu dữ liệu long.

---

#### 2.2. Padding với collate_fn

- Các câu có **độ dài khác nhau**: 5 từ, 10 từ, 20 từ, ...
- DataLoader cần tạo batch → cần **padding** về cùng độ dài

Hàm collate_fn nhận vào một batch các mẫu, tách riêng các sequences của từ và nhãn. Sau đó sử dụng pad_sequence để đưa tất cả các sequences về cùng độ dài, với batch_first=True để batch dimension ở vị trí đầu tiên. Padding được thực hiện bằng PAD_WORD_ID cho từ và PAD_TAG_ID cho nhãn. Hàm trả về hai tensor đã được padding: word_padded và tag_padded.

---

#### 2.3. DataLoader

Tạo hai DataLoader: train_loader với batch_size=32 và shuffle=True để xáo trộn dữ liệu mỗi epoch, và valid_loader với batch_size=32 nhưng shuffle=False để giữ nguyên thứ tự. Cả hai đều sử dụng collate_fn để xử lý padding cho các batch.

**Luồng dữ liệu:**
```
CoNLL-2003 → NERDataset → DataLoader + collate_fn → Batches
```

---

### Task 3: Xây dựng Mô hình Bidirectional LSTM

**Kiến trúc:**
```
Input: [2, 3, 4, 5, 0]  (batch_size=1, seq_len=5)
    ↓
Embedding Layer (vocab_size=23624, embedding_dim=128)
    ↓
embeddings: (1, 5, 128)
    ↓
Bidirectional LSTM (input_size=128, hidden_size=256)
    ↓
lstm_out: (1, 5, 512)  # 256 * 2 (bidirectional)
    ↓
Linear Layer (512 → 10)
    ↓
tag_scores: (1, 5, 10)  # Điểm số cho 10 tags
```

**Các tham số:**
- **vocab_size**: 23,624 (số lượng từ trong vocabulary)
- **embedding_dim**: 128 (chiều của word embeddings)
- **hidden_dim**: 256 (chiều của hidden state LSTM mỗi hướng)
- **num_tags**: 10 (9 NER tags + PAD)
- **bidirectional**: True (xử lý cả 2 chiều)
- **dropout**: 0.5 (giảm overfitting)

**Ưu điểm Bidirectional LSTM:**
- Xử lý ngữ cảnh từ **cả 2 hướng**: trái → phải và phải → trái
- Hiểu rõ hơn về ngữ cảnh xung quanh mỗi từ
- Đặc biệt hiệu quả cho NER vì thực thể phụ thuộc vào cả context trước và sau

Lớp LSTMForNER kế thừa từ nn.Module với ba lớp chính:

**Lớp Embedding:** Chuyển đổi word indices thành vector embeddings với kích thước vocab_size × embedding_dim.

**Lớp LSTM:** Sử dụng Bidirectional LSTM với input_size=embedding_dim, hidden_size=hidden_dim, dropout=0.5 để giảm overfitting, và batch_first=True. Vì là bidirectional nên output có dimension gấp đôi hidden_dim.

**Lớp Linear:** Ánh xạ từ hidden state (kích thước hidden_dim × 2) sang số lượng nhãn NER.

Trong hàm forward, đầu vào sentences được đưa qua embedding layer, sau đó qua LSTM để xử lý chuỗi, cuối cùng qua linear layer để tính điểm số cho mỗi nhãn ở mỗi vị trí.

---

**Khởi tạo mô hình:**

Mô hình LSTMForNER được khởi tạo với vocab_size lấy từ từ điển, embedding_dim=128, hidden_dim=256, và num_tags là số lượng nhãn NER.

Optimizer sử dụng Adam với learning rate 0.001 để tối ưu hóa các tham số của model.

Hàm loss sử dụng CrossEntropyLoss với ignore_index=PAD_TAG_ID để bỏ qua các vị trí padding khi tính loss.

**Loss function:**
- **CrossEntropyLoss**: Tính loss cho bài toán phân loại đa lớp
- **ignore_index=PAD_TAG_ID**: Bỏ qua các vị trí padding khi tính loss

---

### Task 4: Huấn luyện Mô hình

#### Vòng lặp huấn luyện

**Các bước trong 1 epoch:**
1. **Forward pass**: Tính tag_scores từ input
2. **Compute loss**: So sánh dự đoán với thực tế (bỏ qua padding)
3. **Backward pass**: Tính gradients
4. **Update weights**: Optimizer cập nhật tham số

---

#### Kết quả huấn luyện

**Training process (5 epochs):**

```
Epoch 1 | Loss = 0.4993 | Train = 0.9245 | Valid = 0.9090
Epoch 2 | Loss = 0.2093 | Train = 0.9653 | Valid = 0.9333
Epoch 3 | Loss = 0.1067 | Train = 0.9868 | Valid = 0.9438
Epoch 4 | Loss = 0.0491 | Train = 0.9955 | Valid = 0.9476
Epoch 5 | Loss = 0.0208 | Train = 0.9990 | Valid = 0.9469
```

**Phân tích:**
- **Loss giảm mạnh**: 0.4993 → 0.0208 (giảm ~96%)
- **Train accuracy tăng**: 92.45% → 99.90% (cải thiện 7.45%)
- **Valid accuracy tăng**: 90.90% → 94.69% (cải thiện 3.79%)

**Nhận xét:**
- **Epoch 1**: Model học nhanh, accuracy đã đạt >90%
- **Epoch 2-3**: Cải thiện đáng kể, valid accuracy vượt 94%
- **Epoch 4-5**: Train accuracy gần 100%, valid accuracy ổn định ~94.7%

---

### Task 5: Đánh giá Mô hình

#### 5.1. Metrics đánh giá

**Sử dụng thư viện seqeval:**
- **Accuracy**: Tỷ lệ tokens được gán nhãn đúng (bỏ qua padding)
- **Precision**: Độ chính xác của các thực thể dự đoán
- **Recall**: Tỷ lệ thực thể thực sự được tìm ra
- **F1-score**: Trung bình giữa của Precision và Recall


Hàm đánh giá hoạt động như sau: Đặt model ở chế độ eval và khởi tạo danh sách all_preds, all_labels. Trong torch.no_grad() để tắt gradient computation, duyệt qua từng batch trong data_loader. Với mỗi batch, tính outputs từ model, lấy predicted IDs bằng argmax. Sau đó áp dụng mask để loại bỏ các vị trí padding: tạo mask từ điều kiện gold_seq khác tag_pad_id, áp dụng mask cho cả pred_seq và gold_seq. Chuyển các IDs đã mask thành tag strings dựa theo id2tag mapping, thêm vào all_preds và all_labels. Cuối cùng, sử dụng seqeval để tính precision, recall, f1 score và trả về các metrics này cùng với accuracy.

---

#### 5.2. Kết quả đánh giá

**Validation set:**
```
Accuracy:  0.9469 (94.69%)
Precision: 0.7675 (76.75%)
Recall:    0.7023 (70.23%)
F1-score:  0.7335 (73.35%)
```

**Test set:**
```
Accuracy:  0.9267 (92.67%)
Precision: 0.6851 (68.51%)
Recall:    0.6160 (61.60%)
F1-score:  0.6487 (64.87%)
```

**So sánh Valid vs Test:**
- Accuracy giảm: 94.69% → 92.67% (-2%)
- F1 giảm: 73.35% → 64.87% (-8.5%)
- Model tổng quát khá tốt nhưng vẫn còn chênh lệch

**Lý do F1 thấp hơn Accuracy:**
- Accuracy tính ở mức token
- F1 tính ở mức entity (phải đúng toàn bộ entity)
- Ví dụ: "Barack Obama" → nếu dự đoán "Barack" đúng nhưng "Obama" sai thì entity sai

---

#### 5.3. Dự đoán câu mới

**Câu 1: "VNU University is located in Hanoi"**
```
Từ              Nhãn NER       
VNU             B-ORG           (Đúng - tên tổ chức)
University      I-ORG           (Đúng - phần của tổ chức)
is              O               (Đúng - động từ)
located         O               (Đúng - động từ)
in              O               (Đúng - giới từ)
Hanoi           B-LOC           (Đúng - tên địa điểm)
```

**Câu 2: "Barack Obama was born in Hawaii"**
```
Từ              Nhãn NER       
Barack          I-ORG           (Sai - phải là B-PER)
Obama           I-ORG           (Sai - phải là I-PER)
was             O               (Đúng - động từ)
born            O               (Đúng - động từ)
in              O               (Đúng - giới từ)
Hawaii          B-LOC           (Đúng - tên địa điểm)
```

**Câu 3: "Apple Inc is based in Cupertino California"**
```
Từ              Nhãn NER       
Apple           B-ORG           (Đúng - tên công ty)
Inc             I-ORG           (Đúng - phần của tổ chức)
is              O               (Đúng - động từ)
based           O               (Đúng - động từ)
in              O               (Đúng - giới từ)
Cupertino       B-LOC           (Đúng - địa điểm)
California      I-LOC           (Đúng - phần của địa điểm)
```

**Câu 4: "The European Union was founded in 1993"**
```
Từ              Nhãn NER       
The             O               (Đúng - mạo từ)
European        B-ORG           (Đúng - tên tổ chức)
Union           I-ORG           (Đúng - phần của tổ chức)
was             O               (Đúng - động từ)
founded         O               (Đúng - động từ)
in              O               (Đúng - giới từ)
1993            O               (Đúng - số)
```

---

**Phân tích kết quả dự đoán:**

**Dự đoán đúng:**
- **Tổ chức**: "VNU University", "Apple Inc", "European Union"
- **Địa điểm**: "Hanoi", "Cupertino California", "Hawaii"
- **Tokens O**: Động từ, giới từ, mạo từ

**Dự đoán sai:**
- **"Barack Obama"**: Nhầm thành I-ORG thay vì B-PER/I-PER

**Độ chính xác tổng thể:**
- 25/26 tokens đúng = **96.15% accuracy** trên 4 câu test
- 3/4 entities hoàn toàn đúng = **75% entity-level accuracy**

---

## Kết luận

### Những điều đã học được:

1. **Named Entity Recognition**:
   - Bài toán sequence labeling phức tạp hơn POS tagging
   - Lược đồ BIO giúp xác định ranh giới thực thể
   - Cần phân biệt giữa độ chính xác của **token-level** vs **entity-level**

2. **Bidirectional LSTM**:
   - Xử lý ngữ cảnh từ cả 2 hướng
   - Hiệu quả cho NER vì thực thể phụ thuộc vào context xung quanh

3. **Metrics và Đánh giá**:
   - Sử dụng seqeval cho entity-level metrics
   - F1-score quan trọng hơn accuracy cho NER
   - Precision/Recall cung cấp thông tin chi tiết về lỗi
---

## Kết quả cuối cùng

### Độ chính xác trên tập validation:
- **Accuracy**: 94.69% (token-level)
- **Precision**: 76.75%
- **Recall**: 70.23%
- **F1-score**: 73.35%

### Ví dụ dự đoán câu mới:

**Câu: "Apple Inc is based in Cupertino California"**

**Dự đoán:**
```
Apple       → B-ORG  (tổ chức)
Inc         → I-ORG  (tổ chức)
is          → O      (ngoài thực thể)
based       → O      (ngoài thực thể)
in          → O      (ngoài thực thể)
Cupertino   → B-LOC  (địa điểm)
California  → I-LOC  (địa điểm)
```

**Đánh giá**: Model nhận dạng chính xác tổ chức "Apple Inc" và địa điểm "Cupertino California" với lược đồ BIO đúng.

---
