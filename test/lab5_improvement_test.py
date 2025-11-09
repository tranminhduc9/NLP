from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, IDF, HashingTF, RegexTokenizer, SQLTransformer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, regexp_replace, lower, trim
from pyspark.ml.classification import GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def create_baseline_pipeline() -> Pipeline:
    """
    Baseline pipeline sử dụng pipeline của lab5_spark_sentiment_analysis.py
    """
     # Tiền xử lý văn bản
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    # Khởi tạo mô hình Logistic Regression
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

    return pipeline

def preprocess_text_dataframe(df, input_col="text", output_col="cleaned_text"):
    """Tiền xử lý văn bản sử dụng Spark SQL functions"""
    df = df.withColumn(output_col, lower(col(input_col)))
    df = df.withColumn(output_col, regexp_replace(col(output_col), r"http\S+|www\S+|https\S+", ""))
    df = df.withColumn(output_col, regexp_replace(col(output_col), r"<.*?>", ""))
    df = df.withColumn(output_col, regexp_replace(col(output_col), r"[^a-zA-Z\s]", " "))
    df = df.withColumn(output_col, regexp_replace(col(output_col), r"\s+", " "))
    df = df.withColumn(output_col, trim(col(output_col)))
    return df


def create_improved_pipeline() -> Pipeline:
    """
    Pipeline sử dụng Word2Vec với mô hình GBTClassifier
    """
    # Tiền xử lý văn bản
    tokenizer = RegexTokenizer(
        inputCol="cleaned_text",
        outputCol="words",
        pattern=r"\w+|[^\w\s]",  
        gaps=False,
        toLowercase=True
    )
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=2000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    # Khởi tạo mô hình GBTClassifier
    gbt = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, gbt])

    return pipeline

def create_neural_networks_pipeline() -> Pipeline:
    """
    Pipeline sử dụng mạng Neural
    """
    tokenizer = RegexTokenizer(
        inputCol="cleaned_text",
        outputCol="words",
        pattern=r"\w+|[^\w\s]",  
        gaps=False,
        toLowercase=True
    )
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    # Input (5000) → Hidden1 (64) → Hidden2 (32) → Output (2)
    layers = [5000, 64, 32, 2]
    # Khởi tạo mô hình MultilayerPerceptronClassifier
    mlp = MultilayerPerceptronClassifier(
        maxIter=150,
        layers=layers,
        blockSize=128,
        seed=42,
        featuresCol="features",
        labelCol="label"
    )
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, mlp])
    return pipeline

def evaluate_model(predictions) -> dict:
    """Đánh giá mô hình"""
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    
    return {
        'accuracy': evaluator.setMetricName("accuracy").evaluate(predictions),
        'precision': evaluator.setMetricName("weightedPrecision").evaluate(predictions),
        'recall': evaluator.setMetricName("weightedRecall").evaluate(predictions),
        'f1_score': evaluator.setMetricName("f1").evaluate(predictions)
    }

def main():
    # Đường dẫn tới dữ liệu
    data_path = "data/sentiments.csv"
    #Khởi tạo Spark session
    print('Khởi tạo Spark session')
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    # Đọc dữ liệu CSV
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Tiền xử lý
    df = preprocess_text_dataframe(df, input_col="text", output_col="cleaned_text")
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

    # Loại bỏ null trong text và label
    df = df.dropna(subset=["text", "label"])

    df = df.withColumn("label", col("label").cast("double"))
    
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=912)

    # Baseline_model
    print('Tạo Baseline_model...')
    baseline_pipeline = create_baseline_pipeline()

    baseline_model = baseline_pipeline.fit(train_data)

    baseline_predictions = baseline_model.transform(test_data)
    baseline_results = evaluate_model(baseline_predictions)

    # Improved_model
    print('Tạo Improved_model...')
    improved_pipeline = create_improved_pipeline()
    improved_model = improved_pipeline.fit(train_data)

    improved_predictions = improved_model.transform(test_data)
    improved_results = evaluate_model(improved_predictions)

    # Neural_networks_model
    print('Tạo Neural_networks_model...')
    nn_pipeline = create_neural_networks_pipeline()
    nn_model = nn_pipeline.fit(train_data)
    nn_predictions = nn_model.transform(test_data)
    nn_results = evaluate_model(nn_predictions)

    # Hiển thị kết quả
    print('KẾT QUẢ SO SÁNH CÁC MÔ HÌNH')
    print(f"{'Metric':<15} {'Baseline':<20} {'Improved':<20} {'Neural Network':<20}")
    print('-'*80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        print(f"{metric:<15} "
              f"{baseline_results[metric]:<20.4f} "
              f"{improved_results[metric]:<20.4f} "
              f"{nn_results[metric]:<20.4f}"
            )

    spark.stop()


if __name__ == "__main__":
    main()