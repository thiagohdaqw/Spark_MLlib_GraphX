from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
from datetime import datetime

TRAINING_FILE = "/home/d/unb/2022.2/pspd/p3/clean_IMDB Dataset.csv"
MODEL_PATH = "/home/d/unb/2022.2/pspd/p3/"

spark = SparkSession \
    .builder \
    .appName("P2 - PSPD - MLlib - train") \
    .getOrCreate()

# Abrindo o dataset de treinamento
training = spark \
    .read \
    .format("csv") \
    .option("sep", ";") \
    .option("header", "true") \
    .load(TRAINING_FILE) \
    .selectExpr("review as sentence", "CAST(sentiment AS FLOAT) AS label")

# Definindo a arquitetura do modelo
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression()
lrparamGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10, 20, 50])
             .build())
lrevaluator = RegressionEvaluator(metricName="rmse")
lrcv = CrossValidator(estimator = lr,
                    estimatorParamMaps = lrparamGrid,
                    evaluator = lrevaluator,
                    numFolds = 5)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lrcv])


# Executando o treinamento
model = pipeline.fit(training)


# Salvando o modelo em disco
model.save(F"{MODEL_PATH}{datetime.now()}_IMDB.model")