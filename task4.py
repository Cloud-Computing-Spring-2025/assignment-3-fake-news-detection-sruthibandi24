from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import ast

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsClassifier_Task4").getOrCreate()

# Load cleaned data from task2_output.csv
df = spark.read.csv("task2_output.csv", header=True, inferSchema=True)

# UDF to parse the filtered_words string into array
parse_udf = udf(lambda x: ast.literal_eval(x), ArrayType(StringType()))
df = df.withColumn("filtered_words_array", parse_udf(df["filtered_words"]))

# Apply HashingTF
hashingTF = HashingTF(inputCol="filtered_words_array", outputCol="raw_features", numFeatures=10000)
df_tf = hashingTF.transform(df)

# Apply IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

# Label indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf)

# Split into train/test
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

# Train Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

# Predict
predictions = model.transform(test_data)

# Save predictions
predictions.select("id", "title", "label_index", "prediction") \
    .toPandas().to_csv("task4_output.csv", index=False)

# Stop Spark session
spark.stop()