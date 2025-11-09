from sklearn.model_selection import train_test_split
from src.models.text_classifier import TextClassifier
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.tfidf_vectorizer import TfidfVectorizer

def main():
    # Dữ liệu mẫu
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

    # Khởi tạo tokenizer, vectorizer và classifier
    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer)
    classifier = TextClassifier(vectorizer)

    # Chia train, test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    print(f'Dữ liệu train: \n {X_train}')
    print(f'Dữ liệu test: \n {X_test}')

    # Huấn luyện mô hình và dự đoán
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f'Dự đoán nhãn {y_pred}')
    print(f'Nhãn thật {y_test}')
    # Đánh giá mô hình
    results = classifier.evaluate(y_test, y_pred)
    print("Kết quả đánh giá:\n", results)

if __name__ == "__main__":
    main()

