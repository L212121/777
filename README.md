# 777
#load libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline


# initialize Spark
spark = SparkSession.builder.appName("Delay prediction").getOrCreate()

# read data
# we use carrier on-time reporting of January 2024 as our training and testing data
df = spark.read.csv("ONTIME_REPORTING_202401.csv", header=True, inferSchema=True)

# we define a delay(1) as a total delay time exceeding 15 minutes, while others are considered as no delay(0)
df = df.withColumn("is_delayed", when(col("CARRIER_DELAY") + col("WEATHER_DELAY") + col("NAS_DELAY") +
                                      col("SECURITY_DELAY") + col("LATE_AIRCRAFT_DELAY") > 15, 1).otherwise(0))

# fill null values
fill_values = {'AIR_TIME': 0, 'DEP_DELAY': 0, 'ARR_DELAY': 0, 'OP_CARRIER_AIRLINE_ID': 'Unknown'}
df = df.fillna(fill_values)


# logistic regression

# feature transformation
indexer = StringIndexer(inputCols=["ORIGIN_STATE_ABR", "DEST_STATE_ABR", "TAIL_NUM", "DAY_OF_WEEK"],
                        outputCols=["origin_indexed", "dest_indexed", "tail_number_indexed", "day_of_week_indexed"],
                        handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["origin_indexed", "dest_indexed", "tail_number_indexed", "day_of_week_indexed"],
                        outputCols=["origin_encoded", "dest_encoded", "tail_number_encoded", "day_of_week_encoded"])
assembler = VectorAssembler(inputCols=["origin_encoded", "dest_encoded", "tail_number_encoded", "DEP_DELAY",
                                       "AIR_TIME", "day_of_week_encoded"], outputCol="features")

# split dataset
(train_data, test_data) = df.randomSplit([0.7, 0.3], seed=7)

# train the model
lr = LogisticRegression(featuresCol="features", labelCol="is_delayed")
pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])
model = pipeline.fit(train_data)

# prediction
predictions = model.transform(test_data)

# evaluate model
evaluator = BinaryClassificationEvaluator(labelCol="is_delayed")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print(f"Accuracy: {accuracy}")

# stop Spark
spark.stop()
