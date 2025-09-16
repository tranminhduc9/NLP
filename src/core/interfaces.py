from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Split the input text into a list of tokens.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        pass

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, tokens: List[str]) -> None:
        """
        Convert a list of tokens into a numerical vector representation.

        Args:
            tokens (List[str]): The list of tokens to vectorize.

        Returns:
            List[float]: A numerical vector representation of the tokens.
        """
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transform documents into count vectors using the learned vocabulary.

        Args:
            documents (List[str]): List of text documents.

        Returns:
            List[List[int]]: List of count vectors.
        """
        pass

    @abstractmethod
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Learn the vocabulary from the corpus and transform the same corpus
        into count vectors.

        Args:
            corpus (List[str]): List of text documents.

        Returns:
            List[List[int]]: List of count vectors.
        """
        pass