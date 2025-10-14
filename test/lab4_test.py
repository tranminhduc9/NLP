from src.representations.word_embedder import WordEmbedder

def main():
    embedder = WordEmbedder()

    word1 = "king"
    word2 = "queen"

    print(f"1. Vector của từ '{word1}':\n {embedder.get_vector(word1)}\n")

    word3 = "man"
    print(f"2. Độ tương đồng giữa '{word1}' và '{word3}': {embedder.get_similarity(word1, word3)}\n"
          f"   Độ tương đồng giữa '{word1}' và '{word2}': {embedder.get_similarity(word1, word2)}\n")


    most_similar_words = embedder.get_most_similar("computer", topn=10)
    print(f"3. Từ tương tự với 'computer':\n {most_similar_words}")

    document = "The queen rules the country."
    doc_vector = embedder.embed_document(document)
    print(f"4. Vector biểu diễn tài liệu: {document}\n {doc_vector}")

if __name__ == "__main__":
    main()