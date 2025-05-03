from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("FakeNewsClassifier").getOrCreate()

# Load CSV
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Create temporary view
df.createOrReplaceTempView("news_data")

# Show first 5 rows
df.show(5)

# Count total number of articles
print("Total Articles:", df.count())

# Show distinct labels
df.select("label").distinct().show()

# Save first 5 rows to CSV
df.limit(5).toPandas().to_csv("task1_output.csv", index=False)

# Stop Spark session
spark.stop()