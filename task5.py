from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsClassifier_Task5").getOrCreate()

# Load predictions
predictions = spark.read.csv("task4_output.csv", header=True, inferSchema=True)

# Evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy"
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1"
)

# Evaluate
accuracy = accuracy_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Save to CSV
pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score"],
    "Value": [round(accuracy, 2), round(f1_score, 2)]
}).to_csv("task5_output.csv", index=False)

# Stop Spark session
spark.stop()