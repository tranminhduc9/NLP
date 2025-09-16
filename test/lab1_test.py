from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.interfaces import Tokenizer
from src.core.dataset_loaders import load_raw_text_data
from typing import List

DATA_PATH = "./data/UD_English-EWT/en_ewt-ud-train.txt"

def test_tokenizers(tokenizer: Tokenizer, text: List[str]) -> None:
    print(f"Runing tests {tokenizer.__class__.__name__}")
    for txt in text:
        tokens = tokenizer.tokenize(txt)
        print(f"Input: {txt}")
        print(f"Tokens: {tokens}")
        print("-" * 20)

def main():
    sentences = ["Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",]
    
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    test_tokenizers(simple_tokenizer, sentences)
    test_tokenizers(regex_tokenizer, sentences)

    raw_text = load_raw_text_data(DATA_PATH)

    sample_text = raw_text[:500]

    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample (first 100 chars): {sample_text[:100]}...\n")

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")

if __name__ == "__main__":
    main()
