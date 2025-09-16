from src.core.interfaces import Vectorizer, Tokenizer
from typing import List, Dict


class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}
    

    def fit(self, corpus: List[str]) -> None:
        vocab = set()
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            vocab.update(tokens)
        
        vocab = sorted(vocab)

        self.vocabulary_ = {token: idx for idx, token in enumerate(vocab)}

    def transform(self, documents: List[str]) -> List[List[int]]:
        vectors = []

        for doc in documents:
            vector = [0] * len(self.vocabulary_)

            tokens = self.tokenizer.tokenize(doc)

            for token in tokens:
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    vector[index] += 1
            vectors.append(vector)

        return vectors
            
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        self.fit(corpus)
        return self.transform(corpus)
