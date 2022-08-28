from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import split, explode, upper, col
from graphframes import GraphFrame

KAFKA_SERVER = "localhost:9093"
GRAPHS_TOPIC = "graphs"
SEPARATOR = "\s*;\s*"


spark = SparkSession \
    .builder \
    .appName("P2 - PSPD - GraphX") \
    .config(
        "spark.jars.packages", 
        "graphframes:graphframes:0.8.2-spark3.2-s_2.12,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0" # Requires Spark 3.2
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


def foreach_batch_func(lines: DataFrame, _):
    """Do graph processing in a dataframe"""

    # Graph Initialization
    vertices = lines \
        .select(
            explode(
                split(lines.value, SEPARATOR)
            ).alias("name")
        ) \
        .select(upper(col("name")).alias("id"), upper(col("name")).alias("vertice")) \
        .distinct()

    edges = lines \
        .select(
            split(lines.value, SEPARATOR).alias("vertices")
        ) \
        .select(upper(col("vertices")[0]).alias("src"), upper(col("vertices")[1]).alias("dst")) \
        .distinct()

    graph = GraphFrame(vertices, edges)

    # Processing with pagerank
    pageRank = graph.pageRank(resetProbability=0.15, maxIter=5)

    # Visualization
    pageRank.vertices.select("id", "pagerank") \
        .orderBy(col("pageRank").desc()) \
        .write \
        .format("console") \
        .save()

    pageRank.edges.select("src", "dst", "weight") \
        .orderBy(col("weight").desc()) \
        .write \
        .format("console") \
        .save()

    graph.vertices \
        .write \
        .format("console") \
        .save()


# Reading and Writing stream from KAFKA
spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_SERVER) \
    .option("subscribe", GRAPHS_TOPIC) \
    .option("failOnDataLoss", "false") \
    .load() \
    .writeStream \
    .foreachBatch(foreach_batch_func) \
    .option("checkpointLocation", "/tmp/spark/graphs") \
    .trigger(processingTime="10 seconds") \
    .start() \
    .awaitTermination()