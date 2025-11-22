# Lab 6: Transformers và Hugging Face

## Mục tiêu
- Làm quen với thư viện **Hugging Face Transformers**
- Thực hành sử dụng các mô hình Transformer pre-trained
- Hiểu và áp dụng ba kiến trúc Transformer: **Encoder-only**, **Decoder-only**, **Encoder-Decoder**
- Thực hiện các tác vụ NLP: Masked Language Modeling, Text Generation, Sentence Embedding
- So sánh đặc điểm và ứng dụng của từng loại kiến trúc

---

## Tổng quan về Transformers

**Ba kiến trúc chính:**

1. **Encoder-only** (BERT, RoBERTa):
   - Tập trung vào **hiểu ngữ cảnh**
   - Sử dụng bidirectional attention
   - Phù hợp: phân loại, NER, QA

2. **Decoder-only** (GPT, GPT-2, GPT-3):
   - Tập trung vào **sinh văn bản**
   - Sử dụng unidirectional attention
   - Phù hợp: text generation, completion

3. **Encoder-Decoder** (T5, BART):
   - Kết hợp cả encoder và decoder
   - Phù hợp: translation, summarization

---

## Bài 1: Masked Language Modeling (MLM)

### Giới thiệu

**Masked Language Modeling (MLM)** là tác vụ dự đoán các từ bị che giấu (masked) trong câu dựa trên ngữ cảnh xung quanh. Đây là nhiệm vụ pre-training chính của BERT.

**Ví dụ:**
```
Input:  "Hanoi is the [MASK] of Vietnam"
Output: "Hanoi is the capital of Vietnam"
```

### Thực hành

#### 1.1. Khởi tạo Pipeline

Sử dụng Hugging Face pipeline để tự động tải mô hình phù hợp cho tác vụ fill-mask. Pipeline này mặc định sử dụng biến thể của BERT đã được huấn luyện sẵn.

#### 1.2. Dự đoán từ bị MASK

**Câu test:** "Hanoi is the `<mask>` of Vietnam"

**Kết quả top-5 dự đoán:**

| Từ dự đoán | Độ tin cậy | Câu hoàn chỉnh |
|------------|------------|----------------|
| capital | 40.33% | Hanoi is the (capital) of Vietnam |
| Republic | 34.43% | Hanoi is the (Republic) of Vietnam |
| north | 4.17% | Hanoi is the (north) of Vietnam |
| state | 3.12% | Hanoi is the (state) of Vietnam |
| south | 2.46% | Hanoi is the (south) of Vietnam |

**Phân tích:**
- Mô hình dự đoán đúng từ **"capital"** với xác suất cao nhất (40.33%)
- Các lựa chọn khác như "north", "state", "south" có xác suất thấp hơn nhiều

---

#### 1.3. Thử nghiệm với các câu khác

**Câu 1:** "The cat `<mask>` on the mat"
- **Dự đoán:** above
- **Đánh giá:** Không chính xác, đáp án mong đợi là "sat" hoặc "lay"

**Câu 2:** "Yasuo is a champion of `<mask>` of Legends"
- **Dự đoán:** League ✓
- **Đánh giá:** Chính xác

**Câu 3:** "An apple a day keep the `<mask>` away"
- **Dự đoán:** meat
- **Đánh giá:** Sai, đáp án đúng là "doctor" (thành ngữ tiếng Anh)

---

### Câu hỏi và Trả lời

**a. Mô hình dự đoán từ nào với xác suất cao nhất?**

Mô hình dự đoán đúng từ **"capital"** với xác suất cao nhất là **40.33%**.

**b. Vì sao mô hình Encoder-only như BERT phù hợp cho MLM?**

**1. Bidirectional Self-Attention giúp hiểu sâu sắc ngữ cảnh**
- BERT là mô hình mã hóa hai chiều (bidirectional)
- Mỗi token có thể nhìn thấy cả ngữ cảnh bên trái và bên phải
- Điều này rất quan trọng để dự đoán từ bị che `<mask>` dựa trên toàn bộ câu
- Ví dụ: "Hanoi is the [MASK] of Vietnam" → cần nhìn cả "Hanoi" (trái) và "Vietnam" (phải)

**2. Mô hình được huấn luyện sẵn cho nhiệm vụ MLM**
- BERT được pre-train với Masked Language Modeling ngay từ đầu
- Mô hình học cách dự đoán các token bị ẩn trong hàng triệu câu
- Khi sử dụng pipeline "fill-mask", mô hình thực hiện đúng nhiệm vụ đã được học

**3. Encoder-only tập trung vào hiểu ngữ nghĩa**
- MLM chỉ yêu cầu mô hình **hiểu ngữ cảnh** và chọn token phù hợp nhất
- Encoder-only như BERT chuyên về việc "hiểu" văn bản thông qua ngữ cảnh
- Decoder-only như GPT chuyên về "sinh" văn bản tuần tự nên không tối ưu cho MLM

---

## Bài 2: Text Generation

### Giới thiệu

**Text Generation** là tác vụ sinh văn bản tiếp theo dựa trên đoạn prompt.

**Ví dụ:**
```
Prompt: "The best thing about learning NLP is"
Output: "The best thing about learning NLP is that it helps you understand..."
```

### Thực hành

#### 2.1. Khởi tạo Pipeline

Sử dụng pipeline "text-generation" với mô hình GPT-2 mặc định. GPT-2 là mô hình Decoder-only được huấn luyện để dự đoán token tiếp theo.

#### 2.2. Sinh văn bản

**Prompt:** "The best thing about learning NLP is"

**Tham số:**
- `max_length=50`: Tổng độ dài của prompt + văn bản sinh ra
- `num_return_sequences=1`: Số lượng kết quả trả về

**Kết quả:**

```
The best thing about learning NLP is that it's a lot less complex than 
reading, or even writing. It's easy to remember what you're reading, 
and what you're saying.

What's a good NLP?

This question was asked by a few people who've studied NLP. And they've 
all been asked the same thing: What is a good NLP?

I have some pretty good NLP questions, but I've also taken some other 
people's questions and asked them some questions about what they'd like 
to learn in NLP, and what they think is a good NLP...
```

---

### Câu hỏi và Trả lời

**a. Kết quả văn bản được sinh ra có hợp lý không?**

Văn bản sinh ra có **một số ưu điểm và hạn chế**:

**Ưu điểm:**
- Mạch lạc về mặt ngữ pháp
- Câu văn được kết nối tương đối tự nhiên
- Có cấu trúc câu hỏi và giải thích

**Hạn chế:**
- Không trả lời trực tiếp câu hỏi "The best thing about learning NLP is..."
- Nội dung lặp ý và lan man
- Thiếu thông tin cụ thể về lợi ích của việc học NLP
- Chuyển sang đặt câu hỏi thay vì trả lời

**Kết luận:** Văn bản cơ bản hợp lý về mặt ngôn ngữ, nhưng **chưa trả lời đúng trọng tâm** và **chưa tự nhiên**.

**b. Vì sao mô hình Decoder-only (GPT) phù hợp cho tác vụ sinh văn bản?**

Có ba lý do chính:

**1. Kiến trúc tự hồi quy**
- Mô hình dự đoán **token tiếp theo** dựa trên chuỗi token phía trước
- Cơ chế này phù hợp tự nhiên với việc sinh văn bản liên tục
- Mỗi token được sinh ra sẽ trở thành input cho token tiếp theo

**2. Masked Self-Attention một chiều**
- Mỗi token chỉ nhìn được **các token trước nó**, không nhìn các token phía sau
- Điều này ngăn mô hình "gian lận" bằng cách nhìn vào các token sau
- Giúp mô hình sinh văn bản theo đúng trình tự thời gian

**3. Pre-training theo mục tiêu Next-Token Prediction**
- GPT được huấn luyện với mục tiêu dự đoán từ kế tiếp
- Đây chính là nhiệm vụ gốc của sinh văn bản
- Mô hình rất mạnh ở tác vụ này vì đã học trên hàng tỷ ví dụ

**So sánh với Encoder-only:**
- BERT không thể sinh văn bản vì cấu trúc chứa bidirectional
- BERT sẽ "nhìn thấy" toàn bộ câu, không phù hợp với sinh tuần tự

---

## Bài 3: Sentence Embedding

### Giới thiệu

**Sentence Embedding** là quá trình chuyển đổi một câu văn bản thành một vector số có chiều cố định. Vector này nắm bắt được ý nghĩa ngữ nghĩa của câu và có thể dùng cho các tác vụ: Tính độ tương đồng giữa câu, Phân loại văn bản,...

### Thực hành

#### 3.1. Load mô hình BERT

**Mô hình sử dụng:** `bert-base-uncased`
- Kiến trúc Encoder-only
- 12 layers, 768 hidden dimensions
- 110M parameters
- Không phân biệt chữ hoa/thường

#### 3.2. Tokenize câu

**Câu test:** "This is a sample sentence."

**Các bước tokenization:**
1. Tách câu thành các tokens/subwords
2. Thêm các token đặc biệt `[CLS]` và `[SEP]`
3. Chuyển tokens thành IDs
4. Tạo attention_mask để đánh dấu tokens thật vs padding

**Tham số:**
- `padding=True`: Đệm các câu ngắn hơn
- `truncation=True`: Cắt các câu dài quá
- `return_tensors='pt'`: Trả về PyTorch tensors

#### 3.3. Lấy Hidden States

Đưa input qua mô hình BERT để lấy hidden states. Sử dụng `torch.no_grad()` để không tính gradient, tiết kiệm bộ nhớ.

**Output:**
- `last_hidden_state`: Tensor shape `(batch_size, sequence_length, hidden_size)`
- Mỗi token có một vector 768 chiều

#### 3.4. Mean Pooling

**Vấn đề:** Làm thế nào chuyển từ nhiều token embeddings thành một sentence embedding?

**Giải pháp:** Mean Pooling - Tính trung bình của tất cả token embeddings

**Các bước thực hiện:**
1. Lấy `attention_mask` để biết đâu là token thật, đâu là padding
2. Mở rộng attention_mask để khớp với shape của hidden states
3. Nhân hidden states với mask để loại bỏ padding tokens
4. Tính tổng các token embeddings
5. Chia cho số lượng token thật để lấy trung bình

**Kết quả:**
- Vector embedding cuối cùng có shape: `(batch_size, hidden_size)`
- Với `bert-base-uncased`: `(1, 768)`

---

### Câu hỏi và Trả lời

**a. Kích thước của vector biểu diễn và tham số tương ứng trong BERT**

**Kích thước vector:** 768 chiều

**Tham số tương ứng:** `hidden_size` trong cấu hình BERT

**Chi tiết:**
- Vector biểu diễn cuối cùng có kích thước bằng **hidden_size** của mô hình
- Với `bert-base-uncased`: `hidden_size = 768`
- Với `bert-large-uncased`: `hidden_size = 1024`

**Lý do:**
- Mỗi token output từ BERT có dimension = hidden_size
- Sau Mean Pooling, ta lấy trung bình nên vẫn giữ nguyên dimension
- Do đó sentence embedding có cùng dimension với hidden_size

**b. Vì sao cần dùng `attention_mask` khi thực hiện Mean Pooling?**

Có ba lý do quan trọng:

**1. Loại bỏ các token padding**
- Khi batch nhiều câu với độ dài khác nhau, các câu ngắn sẽ được padding bằng token `[PAD]`
- Token `[PAD]` không mang thông tin ngữ nghĩa
- Nếu tính trung bình cả padding → vector embedding bị **sai lệch**

**2. Tính trung bình chính xác**
- `attention_mask` đánh dấu: 1 = token thật, 0 = padding
- Chỉ tính tổng và chia cho số token thật
- Công thức: `mean = sum(embeddings * mask) / sum(mask)`

**3. Đảm bảo tính nhất quán**
- Hai câu giống nhau nhưng padding khác nhau sẽ có embedding giống nhau
- Không bị ảnh hưởng bởi vị trí padding trong batch

**Kết luận:** Sử dụng `attention_mask` giúp sentence embedding **chính xác** và **nhất quán**, không bị ảnh hưởng bởi padding.

---

## Kết luận

### Những điều đã học được:

**1. Hugging Face Transformers:**
- Thư viện mạnh mẽ với hàng nghìn mô hình pre-trained
- Pipeline API giúp sử dụng mô hình dễ dàng
- Hỗ trợ PyTorch và TensorFlow

**2. Masked Language Modeling:**
- BERT dự đoán từ bị mask dựa trên ngữ cảnh hai chiều
- Phù hợp cho các tác vụ cần hiểu sâu về ngữ nghĩa
- Độ chính xác phụ thuộc vào chất lượng pre-training

**3. Text Generation:**
- GPT sinh văn bản theo cách autoregressive
- Chất lượng phụ thuộc vào prompt và hyperparameters
- Cần cẩn thận với hallucination và repetition

**4. Sentence Embedding:**
- Mean Pooling là phương pháp đơn giản nhưng hiệu quả
- Attention mask rất quan trọng để loại bỏ padding
- Sentence embedding dùng cho nhiều downstream tasks

**5. Kiến trúc Transformers:**
- Encoder-only: Hiểu văn bản
- Decoder-only: Sinh văn bản
- Encoder-Decoder: Chuyển đổi văn bản

