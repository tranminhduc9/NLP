from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace

def main():
    """
    Hàm chính để huấn luyện mô hình Word2Vec trên dữ liệu mẫu C4
    và tìm các từ đồng nghĩa.
    """
    # 1. Khởi tạo SparkSession
    # Cấu hình bộ nhớ driver để xử lý các tập dữ liệu lớn hơn
    spark = SparkSession.builder \
        .appName("Word2Vec C4 Training") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # 2. Đọc dữ liệu JSON
    # Thay thế bằng đường dẫn chính xác đến tệp dữ liệu của bạn
    data_path = "data/c4-train.00000-of-01024-30K.json"
    try:
        df = spark.read.json(data_path)
    except Exception as e:
        print(f"Lỗi: Không thể đọc dữ liệu từ đường dẫn '{data_path}'.")
        spark.stop()
        return

    # 3. Tiền xử lý văn bản
    # - Chuyển văn bản thành chữ thường (lowercase)
    # - Loại bỏ các ký tự không phải là chữ cái, số hoặc khoảng trắng để đơn giản hóa
    print("Đang tiền xử lý dữ liệu văn bản...")
    df_processed = df.select(
        lower(col("text")).alias("text_lower")
    ).withColumn(
        "text_clean",
        regexp_replace(col("text_lower"), r"[^a-zA-Z0-9\s]", "")
    )

    # 3.1. Sử dụng Tokenizer để tách câu thành các từ
    print("Đang tách câu thành các từ (tokenizing)...")
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
    df_tokenized = tokenizer.transform(df_processed)

    # 4. Huấn luyện mô hình Word2Vec
    # - vectorSize: Kích thước của vector biểu diễn cho mỗi từ.
    # - minCount: Số lần xuất hiện tối thiểu của một từ để được đưa vào từ điển.
    word2vec = Word2Vec(
        vectorSize=100,
        minCount=5,
        inputCol="words",
        outputCol="vector"
    )

    print("Đang huấn luyện mô hình Word2Vec...")
    model = word2vec.fit(df_tokenized)

    # 5. Tìm các từ tương đồng (synonyms)

    target_word = "computer"
    print(f"\nĐang tìm các từ đồng nghĩa cho '{target_word}'...")

    try:
        # Tìm 5 từ có vector gần nhất 
        synonyms = model.findSynonyms(target_word, 5)
        print(f"5 từ tương tự nhất với '{target_word}':")
        synonyms.show()
    except Exception as e:
        print(f"\nKhông tìm thấy từ '{target_word}' trong từ điển của mô hình.")
        print(f"Lý do có thể là từ này không xuất hiện đủ {word2vec.getMinCount()} lần trong dữ liệu mẫu.")
        print(f"Chi tiết lỗi từ Spark: {e}")

    # 6. Dừng SparkSession
    spark.stop()

if __name__ == "__main__":
    main()
