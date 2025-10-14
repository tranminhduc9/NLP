from gensim.models import Word2Vec
from src.preprocessing.regex_tokenizer import RegexTokenizer
from pathlib import Path


# Đọc data từ file theo từng đoạn để tiết kiệm RAM
class StreamSentences:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.tokenizer = RegexTokenizer()

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            buffer = []
            for line in f:
                line = line.strip()
                if not line:  # dòng trống -> hết 1 đoạn
                    if buffer:
                        text = " ".join(buffer)
                        tokens = self.tokenizer.tokenize(text)
                        if tokens:
                            yield tokens
                        buffer = []
                else:
                    buffer.append(line)

            if buffer:
                text = " ".join(buffer)
                tokens = self.tokenizer.tokenize(text)
                if tokens:
                    yield tokens


# Train Word2Vec model và lưu lại
def train_word2vec(data_path, save_path):
    sentences = StreamSentences(data_path)

    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences, # Dữ liệu huấn luyện
        vector_size=100,    # Số chiều vector từ
        window=5,         # Kích thước window
        min_count=3,    # Số lần xuất hiện tối thiểu 
        workers=4,      # Số luồng xử lý
        sg=1            # 0: CBOW, 1: Skip-gram
    )

    # Lưu mô hình 
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))  
    print(f"Mô hình lưu tại {save_path}")

    return model

# Demo model
def demo_model(model):
    """
    Demo các tác vụ với mô hình Word2Vec đã huấn luyện
    """

    #  Tìm từ gần nghĩa
    target_word = "computer"
    print(f"Tìm từ gần nghĩa với '{target_word}':")
    try:
        similar_words = model.wv.most_similar(target_word, topn=10)
        for word, score in similar_words:
            print(f"  - {word:<15} (Similarity: {score:.4f})")
    except KeyError:
        print(f"  Từ '{target_word}' không có trong từ điển.")

    # Bài toán tương đồng
    print("\n 'king' -> 'man' thì 'queen' -> ?")
    try:
        result = model.wv.most_similar(positive=["king", "man"], negative=["queen"], topn=1)
        answer = result[0][0]
        print(f"  -> Đáp án: '{answer}'")
    except KeyError:
        print(f"  Từ không có trong từ điển")



if __name__ == "__main__":
    data_file = Path("data/UD_English-EWT/en_ewt-ud-train.txt")
    model_file = Path("results/word2vec_ewt.model")

    if not model_file.exists():
        model = train_word2vec(data_file, model_file)
    else:
        print("Loading existing model...")
        model = Word2Vec.load(str(model_file))  

    demo_model(model)
