from typing import List
from src.representations.count_vectorizer import CountVectorizer
from src.preprocessing.regex_tokenizer import RegexTokenizer


def main():
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    regex_tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=regex_tokenizer)

    docterm_matrix = vectorizer.fit_transform(corpus)

    print(docterm_matrix)

if __name__ == "__main__":
    main()