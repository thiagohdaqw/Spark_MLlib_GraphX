from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
from datetime import datetime

# Dataset https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
TRAINING_FILE = "/home/d/unb/2022.2/pspd/p3/clean_IMDB Dataset.csv"
MODEL_PATH = "/home/d/unb/2022.2/pspd/p3/"

spark = SparkSession \
    .builder \
    .appName("P2 - PSPD - MLlib - train") \
    .getOrCreate()

# Prepare training documents from a list of (label, sentence) tuples.
training = spark \
    .read \
    .format("csv") \
    .option("sep", ";") \
    .option("header", "true") \
    .load(TRAINING_FILE) \
    .selectExpr("review as sentence", "CAST(sentiment AS FLOAT) AS label")

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression()
lrparamGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
             .addGrid(lr.maxIter, [5, 25, 50, 100, 150])
             .build())
lrevaluator = RegressionEvaluator(metricName="rmse")
lrcv = CrossValidator(estimator = lr,
                    estimatorParamMaps = lrparamGrid,
                    evaluator = lrevaluator,
                    numFolds = 5)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lrcv])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

model.save(F"{MODEL_PATH}{datetime.now()}_IMDB.model")

test = spark.createDataFrame([
    ("bad",),
    ("i'm not felling very well",),
    ("I'm so happy",),
    ("This is wonderfulll",),
    ("do not know what to do",),
], ["sentence"])

prediction = model.transform(test)
prediction.select("sentence", "probability", "prediction").show()