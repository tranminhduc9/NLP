import gensim.downloader as api
import numpy as np
from src.preprocessing.regex_tokenizer import RegexTokenizer

class WordEmbedder:
    def __init__(self, model_name='glove-wiki-gigaword-50'):
        """Khởi tạo WordEmbedder với mô hình đã có hoặc tự tạo."""
        self.model = api.load(model_name)

    def get_vector(self, word: str):
        """Lấy vector biểu diễn của từ."""
        if word in self.model:
            # Trả về vector của từ
            return self.model[word]
        else:
            # Xử lý lỗi nếu từ không có trong từ điển
            print(f"Từ '{word}' không có trong từ điển.")
            return None

    def get_similarity(self, word1: str, word2: str):
        """Tính độ tương đồng giữa hai từ bằng cách tính cosine."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is not None and vec2 is not None:
            # Tính cosine để tìm similarity
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        else:
            # Xử lý lỗi nếu một trong hai từ không có trong từ điển
            print("1 trong 2 từ không có trong từ điển")
            return None
        
    def get_most_similar(self, word: str, topn: int = 10):
        """Lấy topn từ tương tự với từ đã cho."""
        if word in self.model:
            return self.model.most_similar(word, topn=topn)
        else:
            print(f"Từ '{word}' không có trong từ điển.")
            return None
        
    def embed_document(self, document: str):
        """Biểu diễn một tài liệu dưới dạng trung bình các vector từ."""

        # Sử dụng RegexTokenizer từ Lab1 để tách từ và lấy vector của từng từ
        tokens = RegexTokenizer().tokenize(document)
        vectors = [self.get_vector(word) for word in tokens if self.get_vector(word) is not None]

        # Tính vector của doc bằng trung bình vector của các từ (nếu không trả về vector 0)
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            print("Không có từ nào trong tài liệu có trong từ điển.")
            return np.zeros(self.model.vector_size)