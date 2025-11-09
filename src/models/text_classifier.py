from src.core.interfaces import Vectorizer
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassifier:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self._model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

    def fit(self, texts: List[str], labels: List[int]):
        # Tokenize văn bản
        X_train = self.vectorizer.fit_transform(texts)
        #Huấn luyện mô hình
        self._model.fit(X_train, labels)
    
    def predict(self, texts: List[str]) -> List[int]:
        # Tokenize văn bản
        X_test = self.vectorizer.transform(texts)
        # Dự đoán nhãn
        return self._model.predict(X_test)

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        resutls = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
        return resutls