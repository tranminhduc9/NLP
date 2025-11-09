from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import warnings
warnings.filterwarnings('ignore')

def create_pipeline() -> Pipeline:
    # Tiền xử lý văn bản
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    # Khởi tạo mô hình Logistic Regression
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

    return pipeline

def main():
    
    data_path = "data/sentiments.csv"
    #Khởi tạo Spark session
    print('Khởi tạo Spark session')
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    # Đọc dữ liệu CSV
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    # Đổi nhãn từ -1/1 thành 0/1
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    # Loại bỏ null trong sentiment
    initial_row_count = df.count()
    df = df.dropna(subset=["sentiment"])
    # Chia train test
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=912)


    # Tạo pipeline và huấn luyện mô hình
    print('Tạo pipeline và huấn luyện mô hình...')
    pipeline = create_pipeline()
    model = pipeline.fit(train_data)


    # Dự đoán trên tập test và đánh giá
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    results = {
        'accuracy': evaluator.setMetricName("accuracy").evaluate(predictions),
        'precision': evaluator.setMetricName("weightedPrecision").evaluate(predictions),
        'recall': evaluator.setMetricName("weightedRecall").evaluate(predictions),
        'f1_score': evaluator.setMetricName("f1").evaluate(predictions)
    }

    print(f'Kết quả đánh giá mô hình:\n {results}')
    spark.stop()

if __name__ == "__main__":
    main()