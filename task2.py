
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import lower, col

# Create Spark session
spark = SparkSession.builder.appName("FakeNewsClassifier_Task2").getOrCreate()

# Load the CSV (you can use task1 output or the original)
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Lowercase the text
df_lower = df.withColumn("text_lower", lower(col("text")))

# Tokenizer (split text into words)
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
df_words = tokenizer.transform(df_lower)

# Remove stopwords (like "the", "is", "and", etc.)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_cleaned = remover.transform(df_words)

# Select important columns for output
final_df = df_cleaned.select("id", "title", "filtered_words", "label")

# Save output to CSV
final_df.toPandas().to_csv("task2_output.csv", index=False)

# Stop Spark session
spark.stop()
