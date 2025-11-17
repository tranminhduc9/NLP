# Lab 5 Part 1: Introduction to PyTorch

## Mục tiêu
- Làm quen với **PyTorch**, một framework deep learning phổ biến trong NLP và Computer Vision
- Hiểu cách làm việc với **Tensor** - cấu trúc dữ liệu cơ bản trong PyTorch
- Thực hành các phép toán trên Tensor: cộng, nhân, indexing, slicing, reshape
- Tìm hiểu cơ chế **Autograd** để tính đạo hàm tự động
- Xây dựng mô hình đầu tiên với **torch.nn** (Linear layers, Embedding layers, Neural Network modules)

---

## Nhiệm vụ

### Task 1: Làm quen với Tensor

**Tensor là gì?**
- **Tensor** là cấu trúc dữ liệu đa chiều tương tự như NumPy array
- Là thành phần cơ bản trong PyTorch, được tối ưu hóa để tính toán trên GPU
- Hỗ trợ tính đạo hàm tự động (autograd) cho việc huấn luyện mô hình

---

#### Task 1.1: Tạo Tensor

**Các cách tạo Tensor:**

1. **Từ Python list:**
```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

2. **Từ NumPy array:**
```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

3. **Tạo tensor với giá trị đặc biệt:**
```python
# Tensor toàn số 1
x_ones = torch.ones_like(x_data)

# Tensor ngẫu nhiên
x_rand = torch.rand_like(x_data, dtype=torch.float)
```

**Thuộc tính của Tensor:**
- **Shape**: Kích thước của tensor (số hàng, số cột, ...)
- **Dtype**: Kiểu dữ liệu (float32, int64, ...)
- **Device**: Thiết bị lưu trữ (CPU hoặc GPU)

**Kết quả:**
```
Tensor từ list:
 tensor([[1, 2],
        [3, 4]])

Tensor từ NumPy array:
 tensor([[1, 2],
        [3, 4]])

Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.4963, 0.7682],
        [0.0885, 0.1320]])

Shape của tensor: torch.Size([2, 2])
Datatype của tensor: torch.float32
Device lưu trữ tensor: cpu
```

---

#### Task 1.2: Các phép toán trên Tensor

**Phép toán cơ bản:**

1. **Cộng tensor:**
```python
sum_data = x_data + x_data
```

2. **Nhân với số vô hướng:**
```python
multi_data = x_data * 5
```

3. **Nhân ma trận (matrix multiplication):**
```python
multi_transpose_data = x_data @ x_data.T
```

**Kết quả:**
```python
x_data = tensor([[1., 2.],
                 [3., 4.]])

Tổng của data với data:
tensor([[2., 4.],
        [6., 8.]])

Tích của data với 5:
tensor([[ 5., 10.],
        [15., 20.]])

Tích của data với data transpose:
tensor([[ 5., 11.],
        [11., 25.]])
```

**Phân tích:**
- **Cộng tensor**: Thực hiện element-wise addition (cộng từng phần tử tương ứng)
- **Nhân scalar**: Mỗi phần tử trong tensor nhân với 5
- **Nhân ma trận**: `x_data @ x_data.T` = `[[1,2],[3,4]] @ [[1,3],[2,4]]` = `[[5,11],[11,25]]`

---

#### Task 1.3: Indexing và Slicing

**Truy xuất dữ liệu từ Tensor:**

```python
# Lấy hàng đầu tiên
x_data[0]  # tensor([1., 2.])

# Lấy cột thứ 2
x_data[:, 1]  # tensor([2., 4.])

# Lấy giá trị tại hàng 2, cột 2
x_data[1, 1]  # tensor(4.)
```

**Kết quả:**
```
Hàng đầu tiên: tensor([1., 2.])
Cột thứ 2: tensor([2., 4.])
Lấy ra giá trị hàng 2 cột 2: tensor(4.)
```

**Nhận xét:**
- Indexing trong PyTorch tương tự NumPy
- Index bắt đầu từ 0

---

#### Task 1.4: Thay đổi shape

**Reshape tensor với `.view()`:**

```python
x = torch.rand(4, 4)  # Tensor 4×4
x.view(16, 1)         # Reshape thành 16×1
```

**Kết quả:**
```
tensor([[0.3904],
        [0.6009],
        [0.2566],
        [0.7936],
        [0.9408],
        ...
        [0.2823]])
```

**Lưu ý:**
- `.view()` chỉ thay đổi shape, không thay đổi dữ liệu
- Tổng số phần tử phải giữ nguyên (4×4 = 16 = 16×1)

---

### Task 2: Tính đạo hàm với Autograd

**Autograd là gì?**
- **Autograd** (Automatic Differentiation) là cơ chế tính đạo hàm tự động trong PyTorch
- Quan trọng cho việc **backpropagation** trong neural networks
- Tự động theo dõi các phép toán và tính gradient

---

#### Task 2.1: Thực hành với autograd

**Ví dụ cơ bản:**

```python
# Tạo tensor với requires_grad=True
x = torch.ones(1, requires_grad=True)

# Thực hiện phép toán
y = x + 2  # y = x + 2
z = y * y * 3  # z = 3 * (x+2)^2

# Tính đạo hàm dz/dx
z.backward()

# Gradient được lưu trong x.grad
print(x.grad)  # tensor([18.])
```

**Kết quả:**
```
x: tensor([1.], requires_grad=True)
y: tensor([3.], grad_fn=<AddBackward0>)
grad_fn của y: <AddBackward0 object at 0x...>
Đạo hàm của z theo x: tensor([18.])
```

**Giải thích toán học:**
- Hàm số: `z = 3 * (x+2)²`
- Đạo hàm: `dz/dx = 3 * 2 * (x+2) = 6 * (x+2)`
- Với `x=1`: `dz/dx = 6 * 3 = 18` ✓

---

**Lưu ý về `.backward()`:**

1. **Gọi backward nhiều lần:**
```python
z.backward()
print(x.grad)  # tensor([18.])

z.backward()  # LỖI
```

**Nguyên nhân:**
- Sau khi gọi `.backward()`, computational graph bị xóa để tiết kiệm bộ nhớ
- Không thể backward lần 2

2. **Giải pháp - sử dụng `retain_graph=True`:**
```python
z = y * y * 3
z.backward(retain_graph=True)  # Giữ lại graph
z.backward()  
print(x.grad)  # tensor([36.])
```

**Kết quả:**
```
Đạo hàm của z theo x: tensor([36.])
```

**Giải thích:**
- Gradient được **tích lũy** (accumulated): `18 + 18 = 36`
- Cần gọi `x.grad.zero_()` để reset gradient trước mỗi backward

---

### Task 3: Xây dựng Mô hình đầu tiên với torch.nn

**torch.nn là gì?**
- Module cung cấp các **building blocks** để xây dựng neural networks
- Bao gồm các layers: Linear, Conv2d, LSTM, Embedding, ...
- Tất cả đều kế thừa từ `nn.Module`

---

#### Task 3.1: Lớp nn.Linear

**Linear layer (Fully Connected Layer):**
- Thực hiện phép biến đổi tuyến tính: `y = xW^T + b`
- **Input**: Vector n chiều
- **Output**: Vector m chiều

```python
# Khởi tạo Linear layer: 5 chiều -> 2 chiều
linear_layer = torch.nn.Linear(in_features=5, out_features=2)

# Tạo input: 3 mẫu, mỗi mẫu 5 chiều
input_tensor = torch.randn(3, 5)

# Forward pass
output = linear_layer(input_tensor)
```

**Kết quả:**
```
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
Output:
 tensor([[-0.6536, -0.4858],
        [ 0.1682,  0.4278],
        [-0.3129,  0.2841]], grad_fn=<AddmmBackward0>)
```

**Phân tích:**
- Input: `[3, 5]` → 3 samples, mỗi sample có 5 features
- Output: `[3, 2]` → 3 samples, mỗi sample được biến đổi thành 2 features
- `grad_fn=<AddmmBackward0>` → layer này có thể tính gradient

---

#### Task 3.2: Lớp nn.Embedding

**Embedding layer:**
- Chuyển **chỉ số các từ** thành **vectors embedding**
- Quan trọng trong NLP để biểu diễn từ
- **Input**: Chỉ số của từ (integer)
- **Output**: Vector embedding (float)

```python
# Embedding cho từ điển 10 từ, mỗi từ -> vector 3 chiều
embedding_layer = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)

# Input: indices của các từ trong câu
input_indices = torch.LongTensor([1, 5, 0, 8])

# Lấy embeddings
embeddings = embedding_layer(input_indices)
```

**Kết quả:**
```
Input shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Embeddings:
 tensor([[ 0.8042, -0.1383,  1.3188],
        [-0.1690,  0.9774, -0.5588],
        [-0.9818,  0.0255, -0.5456],
        [ 0.4451,  0.7452, -0.1077]], grad_fn=<EmbeddingBackward0>)
```

**Phân tích:**
- Input: `[4]` → 4 word indices
- Output: `[4, 3]` → 4 từ, mỗi từ được biểu diễn bằng vector 3 chiều
- Mỗi hàng tương ứng với embedding của một từ
- Ví dụ: từ index `1` → vector `[0.8042, -0.1383, 1.3188]`

---

#### Task 3.3: Kết hợp thành 1 nn.Module

**Xây dựng Neural Network hoàn chỉnh:**

```python
class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyFirstModel, self).__init__()
        # Định nghĩa các layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()  # Hàm kích hoạt
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, indices):
        # 1. Word indices -> Embeddings
        embeds = self.embedding(indices)
        
        # 2. Embeddings -> Hidden layer + ReLU
        hidden = self.activation(self.linear(embeds))
        
        # 3. Hidden -> Output
        output = self.output_layer(hidden)
        return output
```

**Kiến trúc mô hình:**
```
Input (word indices) 
    ↓
Embedding Layer (vocab_size → embedding_dim)
    ↓
Linear Layer (embedding_dim → hidden_dim)
    ↓
ReLU Activation
    ↓
Output Layer (hidden_dim → output_dim)
    ↓
Output
```

**Test mô hình:**
```python
model = MyFirstModel(
    vocab_size=100, 
    embedding_dim=16, 
    hidden_dim=8, 
    output_dim=2
)

input_data = torch.LongTensor([[1, 2, 5, 9]])  # 1 câu, 4 từ
output_data = model(input_data)
```

**Kết quả:**
```
Model output shape: torch.Size([1, 4, 2])
```

**Phân tích:**
- **Input**: `[1, 4]` → 1 batch, 4 từ
- **Output**: `[1, 4, 2]` → 1 batch, 4 từ, mỗi từ có 2 output features

**Ứng dụng:**
- Mô hình này có thể dùng cho:
  - **Sequence labeling** (POS tagging, NER)
  - **Sentiment analysis** (nếu aggregate outputs)
  - **Text classification** (với pooling layer)

---

# Giới thiệu về Mạng Nơ-ron Hồi quy (RNNS) và Token classification

## Phần 1: Tại sao các mô hình cũ là chưa đủ?

- Các mô hình Bag-of-Words (CountVectorizer, TF-IDF) xem câu chỉ như một “túi từ”, hoàn toàn **mất thông tin về thứ tự từ**.
- Vì vậy, chúng không phân biệt được hai câu như “The cat sat on the mat” và “The mat sat on the cat”.
- Mạng FFN (lan truyền thẳng) xử lý từng từ độc lập nên **không thể ghi nhớ ngữ cảnh**.
- Ngôn ngữ là dữ liệu tuần tự, nghĩa của từ phụ thuộc vào các từ trước/sau.
- Do đó, ta cần một mô hình có thể xử lý chuỗi và lưu trữ thông tin theo từng bước **RNN**.

---

## Phần 2: Giới thiệu Mạng Nơ-ron Hồi quy (RNN)

### Ý tưởng chính

- RNN có một **bộ nhớ ngắn hạn** nhờ **hidden state**.
- Tại mỗi bước thời gian t, RNN nhận đầu vào hiện tại và trạng thái trước đó rồi mới cho ra trạng thái mới.

## Thuật toán và vấn đề của RNN

### Backpropagation Through Time (BPTT)

- Là thuật toán Backpropagation áp dụng cho RNN đã unroll theo chiều thời gian.
- Gradient phải lan truyền qua nhiều bước → dễ gặp hai vấn đề lớn:

#### 1. Vanishing Gradient (Đạo hàm tiêu biến)

- Gradient trở nên rất nhỏ khi truyền ngược qua nhiều bước.
- Mô hình **quên mất thông tin xa**, không học được phụ thuộc dài hạn.

#### 2. Exploding Gradient (Đạo hàm bùng nổ)

- Gradient tăng quá lớn, gây cập nhật trọng số đột ngột.
- Mạng khó hội tụ và dễ mất ổn định.

### Các kiến trúc RNN phổ biến (Giải quyết vấn đề bộ nhớ)

Để vượt qua vấn đề **Vanishing Gradient**, hai kiến trúc RNN tiên tiến hơn được giới thiệu: **LSTM** và **GRU**. Cả hai đều sử dụng **các cổng (gates)** để kiểm soát luồng thông tin và giúp mô hình ghi nhớ tốt hơn các phụ thuộc dài hạn.

---

#### 1. Long Short-Term Memory (LSTM)

**Ý tưởng chính**
- LSTM bổ sung một thành phần quan trọng gọi là **Cell State** – một loại “đường truyền thông tin dài hạn” chạy xuyên suốt chuỗi thời gian.
- Cell State cho phép mô hình giữ lại thông tin từ rất sớm trong chuỗi mà không bị mất dần.
- LSTM dùng **ba cổng** để điều khiển thông tin:
  - **Cổng quên (Forget Gate):** Quyết định phần thông tin cũ nào nên giữ hoặc bỏ.
  - **Cổng đầu vào (Input Gate):** Quyết định thông tin mới nào sẽ được lưu vào bộ nhớ.
  - **Cổng đầu ra (Output Gate):** Điều chỉnh phần bộ nhớ nào được dùng để tạo ra trạng thái ẩn hiện tại.

#### 2. Gated Recurrent Unit (GRU)

**Ý tưởng chính**
- GRU là phiên bản **đơn giản hóa** của LSTM.
- Thay vì tách riêng Cell State và Hidden State, GRU **kết hợp chúng thành một trạng thái duy nhất**.
- GRU chỉ sử dụng **hai cổng**:
  - **Cổng cập nhật (Update Gate):** Điều khiển lượng thông tin cũ được giữ lại.
  - **Cổng đặt lại (Reset Gate):** Quyết định thông tin quá khứ nào cần bỏ đi khi xử lý đầu vào mới.

---

## Phần 3: Bài toán Phân loại Token (Token Classification)

Phân loại Token là nhiệm vụ gán nhãn cho từng từ trong câu, và RNN rất phù hợp vì nó tạo ra đầu ra ở mỗi bước thời gian.

### Các bài toán tiêu biểu

- **POS Tagging:** Gán loại từ (danh từ, động từ, tính từ...).  
  Ví dụ: *She reads a book* → She/PRP, reads/VBZ, a/DT, book/NN.

- **Named Entity Recognition (NER):** Nhận diện thực thể (người, tổ chức, địa điểm...).  
  Ví dụ: *Apple was founded by Steve Jobs in California* → Apple/ORG, Steve Jobs/PER, California/LOC.

- **Dependency Parsing:** Xác định quan hệ ngữ pháp giữa các từ (chủ ngữ, tân ngữ...).  
  Ví dụ: Trong câu *She reads a book*, “reads” là gốc; “She” là chủ ngữ; “book” là tân ngữ.

- **Semantic Role Labeling (SRL):** Xác định vai trò ngữ nghĩa của các thành phần với động từ.  
  Ví dụ: *Mary sold the book to John* → Mary (Agent), the book (Theme), John (Recipient).

---

## Phần 4: Công cụ triển khai thuật toán Deep Learning

Việc lựa chọn công cụ tùy vào giai đoạn dự án: nghiên cứu, triển khai trên hệ thống lớn, hay tối ưu hóa mô hình.

### Tóm tắt theo giai đoạn

- **Nghiên cứu & Phát triển**
  - Môi trường: Máy cá nhân, Google Colab  
  - Ngôn ngữ: Python  
  - Framework: PyTorch, TensorFlow, Keras  
  - Công cụ: Jupyter Notebook, VS Code  
  - Lý do: Linh hoạt, dễ thử nghiệm mô hình, giao diện trực quan.

- **Triển khai trên Big Data**
  - Môi trường: Cụm máy chủ, Spark, Cloud  
  - Ngôn ngữ: Scala, Python  
  - Framework: Spark NLP, BigDL  
  - Công cụ: Spark, MLflow  
  - Lý do: Xử lý văn bản quy mô lớn trên hệ thống phân tán.

- **Tối ưu hóa hiệu suất**
  - Môi trường: GPU, TPU, Intel Xeon  
  - Ngôn ngữ: Python/C++  
  - Framework: ONNX, TensorRT, OpenVINO  
  - Công cụ: Docker, Kubernetes  
  - Lý do: Tăng tốc suy luận, triển khai tối ưu trên phần cứng chuyên dụng.

---

## Phần 5: Thách thức của RNN và Hướng đi tiếp theo

- RNN truyền thống gặp vấn đề **vanishing/exploding gradients**, đặc biệt với chuỗi dài.  
- Khi gradient lan truyền qua nhiều bước, thông tin quan trọng ở đầu chuỗi có thể bị mất.  
- Điều này khiến RNN cơ bản khó học được các phụ thuộc dài hạn.  
- Để giải quyết, các kiến trúc nâng cấp như **LSTM** và **GRU** được phát triển với cơ chế **gates** nhằm kiểm soát thông tin và duy trì trí nhớ lâu hơn.

















