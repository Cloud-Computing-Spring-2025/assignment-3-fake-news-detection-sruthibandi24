
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsClassifier_Task3").getOrCreate()

# Load the preprocessed data from Task 2
df = spark.read.csv("task2_output.csv", header=True, inferSchema=True)

# Convert 'filtered_words' string back to array (since CSV stores it as a string)
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import ast

# UDF to parse stringified list
parse_udf = udf(lambda x: ast.literal_eval(x), ArrayType(StringType()))

df = df.withColumn("filtered_words_array", parse_udf(df["filtered_words"]))

# Apply HashingTF
hashingTF = HashingTF(inputCol="filtered_words_array", outputCol="raw_features", numFeatures=10000)
df_tf = hashingTF.transform(df)

# Apply IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

# Convert label (FAKE/REAL) to numeric (0.0/1.0)
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf)

# Select required columns
df_final.select("id", "filtered_words_array", "features", "label_index") \
    .toPandas().to_csv("task3_output.csv", index=False)

# Stop Spark
spark.stop()
