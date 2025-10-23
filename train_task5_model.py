from pyspark.sql import SparkSession
from pyspark.sql.functions import window, avg, hour, minute, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Initialize Spark
spark = SparkSession.builder.appName("TrainFareTrendModel").getOrCreate()

# Load training data
data = spark.read.csv("training-dataset.csv", header=True, inferSchema=True)

# Aggregate into 5-minute windows and compute avg fare
windowed = data.groupBy(window(col("timestamp"), "5 minutes")) \
               .agg(avg("fare_amount").alias("avg_fare"))

# Create time-based features
train_df = windowed.withColumn("hour_of_day", hour(col("window.start"))) \
                   .withColumn("minute_of_hour", minute(col("window.start")))

# Assemble features
assembler = VectorAssembler(inputCols=["hour_of_day", "minute_of_hour"], outputCol="features")
train_ready = assembler.transform(train_df)

# Train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="avg_fare")
model = lr.fit(train_ready)

# Save trained model
model.save("models/fare_trend_model_v2")

print("✅ Task 5 trend model saved successfully in models/fare_trend_model_v2")

spark.stop()