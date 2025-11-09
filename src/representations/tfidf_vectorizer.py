from src.core.interfaces import Vectorizer, Tokenizer
from typing import List, Dict
import math


class TfidfVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}
        self.num_documents_: int = 0
    
    def fit(self, corpus: List[str]) -> None:
        # Xây từ điển
        vocab = set()
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            vocab.update(tokens)
        
        vocab = sorted(vocab)
        self.vocabulary_ = {token: idx for idx, token in enumerate(vocab)}
        
        # Tính IDF
        self.num_documents_ = len(corpus)
        document_frequency = {token: 0 for token in self.vocabulary_}
        
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in document_frequency:
                    document_frequency[token] += 1

        for token, df in document_frequency.items():
            self.idf_[token] = math.log((1 + self.num_documents_) / (1 + df)) + 1

    def transform(self, documents: List[str]) -> List[List[float]]:
        vectors = []
        
        for doc in documents:
            vector = [0.0] * len(self.vocabulary_)
            tokens = self.tokenizer.tokenize(doc)

            # Tính TF
            term_freq = {}
            for token in tokens:
                if token in self.vocabulary_:
                    term_freq[token] = term_freq.get(token, 0) + 1
            
            # Tính TF-IDF
            for token, freq in term_freq.items():
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    tf = freq  # Raw term frequency
                    idf = self.idf_.get(token, 0)
                    vector[index] = tf * idf
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        self.fit(corpus)
        return self.transform(corpus)
