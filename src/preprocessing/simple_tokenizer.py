import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens: List[str] = []
        current_token = ""

        punctuation = {".", ",", "?", "!"}

        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in punctuation:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char) 
            else:
                current_token += char

        if current_token:
            tokens.append(current_token)

        return tokens
