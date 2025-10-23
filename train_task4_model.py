from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Initialize Spark
spark = SparkSession.builder.appName("TrainFareModel").getOrCreate()

# Load training data
data = spark.read.csv("training-dataset.csv", header=True, inferSchema=True)

# Prepare features
assembler = VectorAssembler(inputCols=["distance_km"], outputCol="features")
train_df = assembler.transform(data)

# Train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
model = lr.fit(train_df)

# Save the trained model
model.save("models/fare_model")

print("✅ Task 4 model saved successfully in models/fare_model")

spark.stop()