# PSPD - Spark Mllib e GraphX

## Pre-requisitos
- Spark 3.2
- Kafka
- Python (numpy, graphframes e pyspark)

# GraphX
## 1. Execução
```
# Altere no graphx.py o KAFKA_SERVER e o WORDS_TOPIC se necessario.

$ $SPARK_HOME/spark-submit graphframes:graphframes:0.8.2-spark3.2-s_2.12,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 $PROJECT_HOME/graphx.py

# Ou com python
$ python $PROJECT_HOME/graphx.py
```
## 2. Uso
```
$ $KAFKA_HOME/bin/kafka-console-producer.sh --topic graphs --bootstrap-server localhost:9093

# Insira os vertices na forma:
#   origem;destino
#   Exemplo: 
#        sao paulo;brasilia
#        brasilia;salvador
#        joao pessoa;brasilia
#        ...
```

# MLlib
## 1. Treinamento (opcional)
```
# Baixe e descompacte o dataset na pasta do projeto: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Limpe o dataset
$ python clean_dataset.py "IMDB Dataset.csv"

# Edite os caminhos TRAINING_FILE e MODEL_PATH em train.py
# Execute o trainamento (pode levar alguns varios minutos)
$ $SPARK_HOME/bin/spark-submit $PROJECT_HOME/train.py
```
## 2. Uso
```
# Edite as constantes MODEL_FILE, KAFKA_SERVER e PREDICT_TOPIC em predict.py

# Inicie a aplicacao
$ $SPARK_HOME/spark-submit org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 $PROJECT_HOME/predict.py

# Insira as frases
$ $KAFKA_HOME/bin/kafka-console-producer.sh --topic predict --bootstrap-server localhost:9093
```