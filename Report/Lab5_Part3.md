# Lab 5 Part 3: RNNs for POS Tagging

## Mục tiêu
- Áp dụng **Recurrent Neural Networks (RNNs)** cho bài toán **POS Tagging** (gán nhãn từ loại)
- Hiểu cách xử lý **sequence labeling** với PyTorch
- Xây dựng pipeline hoàn chỉnh: load data → preprocess → build model → train → evaluate
- Thực hành với **PyTorch Dataset**, **DataLoader**, **padding**, và **masking**
- So sánh với baseline và đánh giá trên tập test

---

## Bài toán POS Tagging

**POS Tagging (Part-of-Speech Tagging):**
- Gán nhãn từ loại (NOUN, VERB, ADJ, ...) cho mỗi từ trong câu
- Là bài toán **sequence labeling**: mỗi token có 1 nhãn

**Ví dụ:**
```
Input:  The quick brown fox jumps over the lazy dog
Output: DET ADJ   ADJ   NOUN VERB  ADP  DET  ADJ  NOUN
```

**Universal POS Tags (UPOS):**
- 17 nhãn chuẩn: NOUN, VERB, ADJ, ADV, DET, ADP, PRON, CONJ, NUM, ...
- Dataset: **UD English-EWT** (Universal Dependencies)

---

## Nhiệm vụ

### Task 1: Chuẩn bị dữ liệu

#### 1.1. Chuẩn bị và tiền xử lý dữ liệu conllu

**Format CoNLL-U:**
```
# sent_id = 1
# text = From the AP comes this story:
1	From	from	ADP	IN	_	3	case	_	_
2	the	the	DET	DT	_	3	det	_	_
3	AP	AP	PROPN	NNP	_	4	obl	_	_
4	comes	come	VERB	VBZ	_	0	root	_	_
5	this	this	DET	DT	_	6	det	_	_
6	story	story	NOUN	NN	_	4	nsubj	_	_
7	:	:	PUNCT	:	_	4	punct	_	SpaceAfter=No

```

**Cột quan trọng:**
- Cột 2: **FORM** (từ)
- Cột 4: **UPOS** (Universal POS tag)

---

#### 1.2. Xây dựng từ điển

**Tạo word_to_ix và tag_to_ix:**

**Kết quả:**
```
Kích thước word_to_ix: 19,675  (từ vựng)
Kích thước tag_to_ix: 17       (UPOS tags)
```

---

### Task 2: Tạo PyTorch Dataset và DataLoader

#### 2.1. POSDataset

**Custom Dataset class:**

Lớp `POSDataset` là một custom Dataset dùng cho bài toán gán nhãn từ loại (POS tagging) trong PyTorch.  
Nó thực hiện các chức năng sau:

- Nhận vào danh sách các câu đã được xử lý dưới dạng:  
  `[[("From", "ADP"), ("the", "DET"), ...], ...]`
- Chuyển từng câu thành hai tensor:
  - **word_indices**: dãy các chỉ số từ (word ID) dựa theo `word_to_ix`
  - **tag_indices**: dãy các chỉ số nhãn (tag ID) dựa theo `tag_to_ix`
- Tự động xử lý từ không có trong từ điển bằng cách thay bằng token **`<UNK>`**
- Hỗ trợ PyTorch DataLoader bằng cách định nghĩa:
  - `__len__()` → trả về số lượng câu
  - `__getitem__()` → trả về (word_tensor, tag_tensor) của một câu

Lớp dataset này giúp DataLoader có thể batch, pad và đưa dữ liệu vào mô hình RNN/LSTM một cách thuận tiện.

---

#### 2.2. Padding với collate_fn

- Các câu có **độ dài khác nhau**: 5 từ, 10 từ, 20 từ, ...
- DataLoader cần tạo batch → cần **padding** về cùng độ dài bằng cách thêm 0 vào phía sau.

---

#### 2.3. DataLoader

- `DataLoader` giúp chia dữ liệu thành từng batch, tự động lấy mẫu, và chuẩn bị dữ liệu cho mô hình.
- `batch_size=32` → mỗi batch gồm 32 câu.
- `shuffle=True` trong train_loader:
  - giúp mô hình học tốt hơn bằng cách xáo trộn dữ liệu mỗi epoch.
- `shuffle=False` trong dev_loader:
  - giữ nguyên thứ tự dữ liệu để đánh giá ổn định.
- `collate_fn=collate_fn`:
  - hàm đặc biệt dùng để **pad các câu** trong batch về cùng độ dài.
  - đảm bảo batch có dạng tensor chuẩn để đưa vào RNN/LSTM.



**Luồng dữ liệu:**
```
Raw data → POSDataset → DataLoader + collate_fn → Batches
```

---

### Task 3: Xây dựng Mô hình RNN


**Kiến trúc:**
```
Input: [2, 3, 4, 5, 0]  (batch_size=1, seq_len=5)
    ↓
Embedding Layer (vocab_size=19675, embedding_dim=128)
    ↓
embeddings: (1, 5, 128)
    ↓
RNN Layer (input_size=128, hidden_size=256)
    ↓
rnn_out: (1, 5, 256)  # Output cho mỗi timestep
    ↓
Linear Layer (256 → 17)
    ↓
tag_scores: (1, 5, 17)  # Điểm số cho 17 tags, mỗi từ
```

**Các tham số:**
- **vocab_size**: 19,675 (số lượng từ trong vocabulary)
- **embedding_dim**: 128 (chiều của word embeddings)
- **hidden_dim**: 256 (chiều của hidden state RNN)
- **num_tags**: 17 (số lượng UPOS tags)

---

**Khởi tạo mô hình:**

**Loss function:**
- **CrossEntropyLoss**: Tính loss khi phân loại
- **ignore_index=PAD_TAG_ID**: Bỏ qua các vị trí padding khi tính loss

---

### Task 4: Huấn luyện Mô hình

#### Vòng lặp huấn luyện

**Các bước trong 1 epoch:**
1. **Forward pass**: Tính tag_scores từ input
2. **Compute loss**: So sánh dự đoán với thực tế
3. **Backward pass**: Tính gradients
4. **Update weights**: Tôi ưu cập nhật tham số

---

#### Kết quả huấn luyện

**Training process (5 epochs):**

```
Epoch 1 | Loss = 0.8771 | Train = 0.8249 | Dev = 0.7921
Epoch 2 | Loss = 0.4740 | Train = 0.8823 | Dev = 0.8461
Epoch 3 | Loss = 0.3462 | Train = 0.9111 | Dev = 0.8620
Epoch 4 | Loss = 0.2687 | Train = 0.9316 | Dev = 0.8734
Epoch 5 | Loss = 0.2127 | Train = 0.9470 | Dev = 0.8846
```

**Phân tích:**
- **Loss giảm đều**: 0.8771 → 0.2127
- **Train accuracy tăng**: 82.49% → 94.70%
- **Dev accuracy tăng**: 79.21% → 88.46%
- **Gap Train-Dev**: ~6% 

**Nhận xét:**
- Epoch 1: Model học nhanh, accuracy nhảy lên ~82%
- Epoch 2-3: Cải thiện đáng kể (~86%)
- Epoch 4-5: Tốc độ cải thiện chậm lại, gần hội tụ

---

### Task 5: Đánh giá Mô hình

#### 5.1. Masking

**Tại sao cần mask?**
- Padding tokens không có ý nghĩa
- Không nên tính vào accuracy/loss
- Tránh model học pattern sai từ padding

---

#### 5.2. Kết quả đánh giá

**Dev set:**
```
Độ chính xác cuối cùng trên tập Dev: 0.8846 (88.46%)
```

**Test set:**
```
Độ chính xác cuối cùng trên tập Test: 0.8815 (88.15%)
```

**Phân tích:**
- **Dev ≈ Test** (88.46% vs 88.15%), Mô hình tổng quát tốt

---

#### 5.3. Dự đoán câu mới


```
test_sentence = "The quick brown fox jumps over the lazy dog"
```

**Kết quả:**
```
Từ              POS Tag   
The             DET       
quick           ADJ       
brown           ADJ       
fox             NOUN      
jumps           VERB      
over            ADV       
the             DET       
lazy            ADJ       
dog             NOUN      
```

**Phân tích:**
-  **The**: DET (Determiner) - Chính xác
-  **quick, brown, lazy**: ADJ (Adjective) - Chính xác
-  **fox, dog**: NOUN - Chính xác
-  **jumps**: VERB - Chính xác
-  **over**: Dự đoán ADV (Adverb) - Sai, là  **ADP** (Adposition/Preposition)

---



