# Lab 7: Dependency Parsing

## Mục tiêu
- Làm quen với **Dependency Parsing** - phân tích cú pháp phụ thuộc
- Thực hành sử dụng thư viện **spaCy** cho phân tích cây phụ thuộc
- Hiểu và truy cập các thành phần trong cây phụ thuộc: token, head, children, dependency relations
- Trực quan hóa cây phụ thuộc bằng **displaCy**
- Duyệt cây phụ thuộc để trích xuất thông tin ngôn ngữ học
- Xây dựng các thuật toán tìm kiếm trên cây phụ thuộc

---

## Tổng quan về Dependency Parsing

**Dependency Parsing** là quá trình phân tích cấu trúc ngữ pháp của câu bằng cách xác định mối quan hệ phụ thuộc giữa các từ. Mỗi từ trong câu có thể phụ thuộc vào một từ khác (gọi là **head**), và mối quan hệ này được gắn nhãn bằng một **dependency relation**.

### Các khái niệm cơ bản:

1. **Token**: Đơn vị từ trong câu
2. **Head**: Từ mà token hiện tại phụ thuộc vào
3. **Dependency Relation (dep_)**: Loại quan hệ phụ thuộc (nsubj, dobj, amod, etc.)
4. **ROOT**: Từ gốc của câu, thường là động từ chính
5. **Children**: Các token phụ thuộc vào token hiện tại

---

## Phần 1: Cài đặt và Khởi tạo

### 1.1. Cài đặt thư viện

Sử dụng thư viện **spaCy** - một trong những thư viện NLP mạnh mẽ nhất cho Python, hỗ trợ phân tích cú pháp phụ thuộc tích hợp sẵn.

**Các bước:**
1. Cài đặt spaCy
2. Tải mô hình ngôn ngữ tiếng Anh `en_core_web_sm`

**Mô hình `en_core_web_sm`:**
- Mô hình nhỏ gọn (~12 MB)
- Hỗ trợ: tokenization, POS tagging, dependency parsing, NER
- Phù hợp cho các tác vụ cơ bản

### 1.2. Load mô hình và phân tích câu

**Câu test đầu tiên:** "The quick brown fox jumps over the lazy dog."

Mô hình spaCy tự động phân tích:
- Tách câu thành tokens
- Gán POS tags
- Xây dựng cây phụ thuộc
- Xác định quan hệ giữa các từ

---

## Phần 2: Trực quan hóa Cây Phụ thuộc

### 2.1. Sử dụng displaCy

**displaCy** là công cụ trực quan hóa tích hợp sẵn trong spaCy, giúp hiển thị cây phụ thuộc dưới dạng đồ họa.

**Tính năng:**
- Hiển thị cây phụ thuộc với các mũi tên cong
- Gán nhãn dependency relations
- Có thể render dưới dạng HTML hoặc SVG

### 2.2. Kết quả trực quan hóa

Khi chạy `displacy.serve(doc, style="dep")`, một server web được khởi chạy tại `http://127.0.0.1:5000` hiển thị cây phụ thuộc.

**Phân tích câu:** "The quick brown fox jumps over the lazy dog."

**Cấu trúc cây:**
- **ROOT**: "jumps" - động từ chính
- **Chủ ngữ (nsubj)**: "fox" phụ thuộc vào "jumps"
- **Bổ nghĩa cho chủ ngữ**: 
  - "The" (det) → "fox"
  - "quick" (amod) → "fox"
  - "brown" (amod) → "fox"
- **Giới từ (prep)**: "over" phụ thuộc vào "jumps"
- **Tân ngữ của giới từ (pobj)**: "dog" phụ thuộc vào "over"
- **Bổ nghĩa cho tân ngữ**:
  - "the" (det) → "dog"
  - "lazy" (amod) → "dog"
- **Dấu câu (punct)**: "." phụ thuộc vào "jumps"

---

## Phần 3: Truy cập Các Thành phần Trong Cây

### 3.1. Thuộc tính của Token

Mỗi token trong spaCy có các thuộc tính quan trọng:

| Thuộc tính | Mô tả | Ví dụ |
|------------|-------|-------|
| `text` | Văn bản gốc của token | "Apple" |
| `pos_` | Part-of-speech tag | "PROPN" |
| `dep_` | Dependency relation | "nsubj" |
| `head` | Token mà nó phụ thuộc vào | "looking" |
| `head.pos_` | POS tag của head | "VERB" |
| `children` | Iterator các token con | ["is"] |
| `i` | Index của token trong câu | 0 |

### 3.2. Ví dụ phân tích

**Câu:** "Apple is looking at buying U.K. startup for $1 billion"

**Kết quả phân tích từng token:**

```
TEXT         | DEP        | HEAD TEXT    | HEAD POS | CHILDREN
----------------------------------------------------------------------
Apple        | nsubj      | looking      | VERB     | []
is           | aux        | looking      | VERB     | []
looking      | ROOT       | looking      | VERB     | ['Apple', 'is', 'at', 'startup', 'for']
at           | prep       | looking      | VERB     | ['buying']
buying       | pcomp      | at           | ADP      | ['U.K.']
U.K.         | npadvmod   | buying       | VERB     | []
startup      | dobj       | looking      | VERB     | []
for          | prep       | looking      | VERB     | ['$']
$            | quantmod   | billion      | NUM      | ['1']
1            | compound   | billion      | NUM      | []
billion      | pobj       | for          | ADP      | ['$']
```

**Nhận xét:**
- "looking" là động từ ROOT của câu
- "Apple" là chủ ngữ (nsubj) của "looking"
- "is" là trợ động từ (aux) của "looking"
- "startup" là tân ngữ trực tiếp (dobj) của "looking"
- Các quan hệ giới từ được thể hiện qua prep, pobj, pcomp

---

## Phần 4: Duyệt Cây Phụ Thuộc để Trích Xuất Thông Tin

### 4.1. Bài toán: Tìm chủ ngữ và tân ngữ của động từ

**Mục tiêu:** Trích xuất các bộ ba (subject, verb, object) từ câu.

**Phương pháp:**
1. Duyệt qua tất cả tokens
2. Tìm các token có POS tag là "VERB"
3. Với mỗi động từ, duyệt qua các children
4. Tìm child có `dep_ == "nsubj"` (chủ ngữ)
5. Tìm child có `dep_ == "dobj"` (tân ngữ trực tiếp)

**Câu test:** "The cat chased the mouse and the dog watched them."

**Kết quả:**
```
Found Triplet: (cat, chased, mouse)
Found Triplet: (dog, watched, them)
```

**Phân tích:**
- Câu có 2 mệnh đề độc lập với 2 động từ
- "chased" có chủ ngữ "cat" và tân ngữ "mouse"
- "watched" có chủ ngữ "dog" và tân ngữ "them"
- Cả hai bộ ba đều được trích xuất chính xác

**Ứng dụng thực tế:**
- Information Extraction: Trích xuất quan hệ từ văn bản
- Question Answering: Trả lời câu hỏi "Who did what to whom?"
- Knowledge Graph Construction: Xây dựng đồ thị tri thức

---

### 4.2. Bài toán: Tìm tính từ bổ nghĩa cho danh từ

**Mục tiêu:** Tìm các tính từ mô tả cho mỗi danh từ trong câu.

**Phương pháp:**
1. Duyệt qua tất cả tokens
2. Tìm các token có POS tag là "NOUN"
3. Với mỗi danh từ, duyệt qua các children
4. Thu thập các child có `dep_ == "amod"` (adjectival modifier)

**Câu test:** "The big, fluffy white cat is sleeping on the warm mat."

**Kết quả:**
```
Danh từ 'cat' được bổ nghĩa bởi các tính từ: ['big', 'fluffy', 'white']
Danh từ 'mat' được bổ nghĩa bởi các tính từ: ['warm']
```

**Phân tích:**
- "cat" có 3 tính từ: "big", "fluffy", "white"
- "mat" có 1 tính từ: "warm"
- Thứ tự tính từ được bảo toàn từ câu gốc

**Ứng dụng thực tế:**
- Sentiment Analysis: Phân tích cảm xúc dựa trên tính từ
- Product Review Analysis: Trích xuất đặc điểm sản phẩm
- Text Summarization: Tóm tắt văn bản dựa trên mô tả quan trọng

---

## Phần 5: Bài Tập Tự Luyện

### Bài 1: Tìm động từ chính của câu

**Đề bài:** Viết hàm `find_main_verb(doc)` để tìm động từ ROOT của câu.

**Nguyên lý:**
- Mỗi câu có duy nhất một token với `dep_ == "ROOT"`
- Token ROOT thường là động từ chính của câu
- Tất cả các token khác trực tiếp hoặc gián tiếp phụ thuộc vào ROOT

**Thuật toán:**
1. Duyệt qua tất cả tokens trong doc
2. Kiểm tra token nào có `dep_ == "ROOT"`
3. Trả về token đó

**Kết quả test:**

| Câu | Động từ chính | POS | Lemma |
|-----|---------------|-----|-------|
| "The quick brown fox jumps over the lazy dog." | jumps | VERB | jump |
| "Apple is looking at buying U.K. startup for $1 billion" | looking | VERB | look |
| "The cat chased the mouse and the dog watched them." | chased | VERB | chase |
| "I love learning Natural Language Processing." | love | VERB | love |

**Nhận xét:**
- Động từ ROOT là động từ chính thể hiện hành động/trạng thái chính của câu
- Với câu có nhiều mệnh đề, ROOT là động từ của mệnh đề chính
- Câu "The cat chased..." có "chased" là ROOT, "watched" là động từ phụ thuộc vào "chased" qua quan hệ "conj"

---

### Bài 2: Trích xuất các cụm danh từ (Noun Chunks)

**Đề bài:** Viết hàm `extract_noun_chunks(doc)` để tự động trích xuất các cụm danh từ.

**Khái niệm Noun Chunk:**
Một cụm danh từ (noun phrase) bao gồm:
- **Danh từ chính (head noun)**: Từ trung tâm của cụm
- **Các từ bổ nghĩa**: Determiner, adjective, compound noun, etc.

**Các loại dependency quan trọng:**
- `det`: Determiner (the, a, an, my, this, etc.)
- `amod`: Adjectival modifier (big, red, beautiful, etc.)
- `compound`: Compound noun modifier (ice cream, school bus, etc.)
- `nummod`: Numeric modifier (three, 100, many, etc.)
- `poss`: Possessive modifier (my, John's, etc.)

**Thuật toán:**
1. Duyệt qua tất cả tokens
2. Tìm các danh từ (NOUN, PROPN, PRON) chưa được xử lý
3. Thu thập các children với dependency thuộc danh sách trên
4. Sắp xếp theo vị trí trong câu
5. Ghép thành cụm danh từ hoàn chỉnh

**So sánh với spaCy:**

**Câu 1:** "The quick brown fox jumps over the lazy dog."

| Phương pháp | Noun Chunks |
|-------------|-------------|
| spaCy | The quick brown fox, the lazy dog |
| Custom | The quick brown fox (root: fox), the lazy dog (root: dog) |

**Câu 2:** "Apple is looking at buying U.K. startup for $1 billion"

| Phương pháp | Noun Chunks |
|-------------|-------------|
| spaCy | Apple, U.K. startup, $ 1 billion |
| Custom | Apple (root: Apple), U.K. (root: U.K.), startup (root: startup), $ 1 billion (root: billion) |

**Câu 3:** "The big fluffy white cat sleeps on the warm comfortable mat."

| Phương pháp | Noun Chunks |
|-------------|-------------|
| spaCy | The big fluffy white cat, the warm comfortable mat |
| Custom | The big fluffy white cat (root: cat), the warm comfortable mat (root: mat) |

**Nhận xét:**
- Hàm custom tương đối chính xác so với spaCy
- spaCy có thuật toán phức tạp hơn, xử lý các trường hợp đặc biệt
- Custom function đơn giản nhưng đủ dùng cho các trường hợp cơ bản
- Cả hai đều giữ nguyên thứ tự các từ bổ nghĩa

**Ứng dụng:**
- Text Mining: Trích xuất khái niệm và thực thể
- Search Engine: Cải thiện độ chính xác tìm kiếm
- Document Classification: Phân loại dựa trên các cụm danh từ chính

---

### Bài 3: Tìm đường đi ngắn nhất trong cây

**Đề bài:** Viết hàm `get_path_to_root(token)` để tìm đường đi từ token bất kỳ lên ROOT.

**Nguyên lý:**
- Cây phụ thuộc là một cây có hướng với ROOT ở đỉnh
- Mỗi token (trừ ROOT) có đúng một head
- Luôn tồn tại đường đi duy nhất từ bất kỳ token nào lên ROOT

**Thuật toán:**
1. Bắt đầu từ token hiện tại
2. Thêm token vào đường đi
3. Di chuyển lên head của token
4. Lặp lại cho đến khi gặp ROOT
5. Trả về danh sách các token trên đường đi

**Câu test:** "The quick brown fox jumps over the lazy dog."

**Kết quả:**

**Token "The" (vị trí 0):**
```
Đường đi: The -> fox -> jumps -> ROOT
Khoảng cách: 2 bước
```

**Giải thích:**
- "The" phụ thuộc vào "fox" (determiner)
- "fox" phụ thuộc vào "jumps" (subject)
- "jumps" là ROOT
- Tổng cộng 2 bước từ "The" đến ROOT

**Token "brown" (vị trí 2):**
```
Đường đi: brown -> fox -> jumps -> ROOT
Khoảng cách: 2 bước
```

**Giải thích:**
- "brown" phụ thuộc vào "fox" (adjective modifier)
- "fox" phụ thuộc vào "jumps" (subject)
- "jumps" là ROOT
- Cùng khoảng cách với "The"

**Token "dog" (vị trí 8):**
```
Đường đi: dog -> over -> jumps -> ROOT
Khoảng cách: 2 bước
```

**Giải thích:**
- "dog" phụ thuộc vào "over" (object of preposition)
- "over" phụ thuộc vào "jumps" (preposition)
- "jumps" là ROOT
- Mọi token ở vị trí lá đều có khoảng cách tương tự

**Phân tích khoảng cách các token:**

| Token | Khoảng cách đến ROOT | Vị trí trong cây |
|-------|---------------------|------------------|
| The | 2 | Lá (determiner) |
| quick | 2 | Lá (modifier) |
| brown | 2 | Lá (modifier) |
| fox | 1 | Nhánh (subject) |
| jumps | 0 | ROOT |
| over | 1 | Nhánh (preposition) |
| the | 3 | Lá (determiner, xa nhất) |
| lazy | 3 | Lá (modifier, xa nhất) |
| dog | 2 | Lá (object) |

**Nhận xét:**
- Các từ gần ROOT có khoảng cách ngắn hơn
- Các từ ở nhánh xa (như "the", "lazy" ở cuối câu) có khoảng cách lớn nhất (3 bước)
- ROOT có khoảng cách = 0

---
