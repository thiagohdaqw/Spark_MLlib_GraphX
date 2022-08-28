from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lower, when, col

MODEL_FILE = "/home/d/unb/2022.2/pspd/p3/IMDB.model"
KAFKA_SERVER = 'localhost:9093'
PREDICT_TOPIC = 'predict'

spark = SparkSession \
    .builder \
    .appName("P2 - PSPD - MLLIB - predict") \
    .config(
        "spark.jars.packages", 
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0"          # Requires Spark 3.2
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


def foreach_batch_func(df: DataFrame, _):
    sentences = df.select(lower(df.value).alias("sentence"))

    model = PipelineModel.load(MODEL_FILE)
    prediction = model.transform(sentences)

    prediction \
        .select(
            "sentence",
            "probability",
            when(col("prediction") == 1.0, "positive").otherwise("negative").alias("prediction")
        ) \
        .write \
        .format("console") \
        .save()


lines = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_SERVER) \
    .option("subscribe", PREDICT_TOPIC) \
    .option("failOnDataLoss", "false") \
    .load() \
    .writeStream \
    .foreachBatch(foreach_batch_func) \
    .option("checkpointLocation", "/tmp/spark/mllib-predict") \
    .trigger(processingTime="10 seconds") \
    .start() \
    .awaitTermination()
