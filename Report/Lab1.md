# Lab 1: Text Tokenization

## Mục tiêu:
 - Mục tiêu của Lab này là hiểu và cài đặt bước tiền xử lý cơ bản trong NLP: Tokenization.
 - **SimpleTokenizer**: dựa trên xử lý chuỗi cơ bản.
 - **RegexTokenizer**: sử dụng biểu thức chính quy để tách từ và dấu câu.

## Cấu trúc dự án:

NLP
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│   └───UD_English-EWT
│           
│
├───Report
│       Lab1.md
│
├───src
│   ├───core
│   │      dataset_loaders.py
│   │      interfaces.py
│   │      __init__.py
│   │
│   ├───preprocessing
│         regex_tokenizer.py
│         simple_tokenizer.py
│         __init__.py
│
└───test
       lab1_test.py
       __init__.py

## Nhiệm vụ:
### Task 1: Cài đặt SimpleTokenizer 
 - Định nghĩa interface Tokenizer trong src/core/interfaces.py với phương thức trừu tượng
 - Tạo lớp SimpleTokenizer kế thừa Tokenizer trong src/core/preprocessing/simple_tokenizer.py 
 - Chuyển văn bản thành chữ thường.
 - Tách từ theo khoảng trắng.
 - Xử lý dấu câu đơn giản bằng cách tách dấu câu khỏi từ.

### Task 2: Cài đặt RegexTokenizer
 - Tạo lớp RegexTokenizer kế thừa Tokenizer trong src/preprocessing/regex_tokenizer.py.
 - Cài đặt phương thức tokenize sử dụng biểu thức chính quy để tách token.
 - Regex được sử dụng:
        + \w+ để bắt các từ hoặc số.
        + [^\w\s] để bắt các ký tự đặc biệt và dấu câu.
 - Cách tiếp cận này giúp tokenizer xử lý văn bản chính xác hơn so với tách theo khoảng trắng.

## Kết quả:
 
### 1. SimpleTokenizer
 - input: ["Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",]

 **Kết quả:**
 ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
 ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
 ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

 - input: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of ...

 **Kết quả**
 ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']

### 2. RegexTokenizer
  - input: ["Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",]

**Kết quả:**
 ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
 ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
 ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

- input: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of ...

**Kết quả:**
['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']

**Phân tích:**
- SimpleTokenizer giữ nguyên các từ nối bằng dấu `-` như `"al-zaman"`, `"al-ani"`.  
- RegexTokenizer tách các dấu `-` ra thành token riêng, gây nhiều token nhỏ hơn.  
- SimpleTokenizer phù hợp hơn với các tên riêng hoặc từ nối, còn RegexTokenizer chuẩn hóa nghiêm khắc hơn theo ký tự và dấu câu.

---

### Kết luận  
- **RegexTokenizer:** chuẩn hóa kỹ, tách mọi ký tự không phải chữ hoặc số, phù hợp xử lý NLP nghiêm ngặt nhưng mất thông tin từ nối/tên riêng.  
- Khi xử lý **UD_English-EWT**, SimpleTokenizer cho token hợp lý hơn cho các thực thể như tên riêng.  

