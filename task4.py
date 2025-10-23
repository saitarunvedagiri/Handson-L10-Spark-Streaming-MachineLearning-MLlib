import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, abs as abs_diff
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# Create Spark Session
spark = SparkSession.builder.appName("Task4_FarePrediction_Assignment").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Define paths for the model and training data
MODEL_PATH = "models/fare_model"
TRAINING_DATA_PATH = "training-dataset.csv"

# --- PART 1: MODEL TRAINING (Offline) ---
# Train the model only if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] No model found. Training a new model using {TRAINING_DATA_PATH}...")

    # Load the training data
    train_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=True)

    # Cast columns to DoubleType
    train_df = train_df_raw.withColumn("distance_km", col("distance_km").cast(DoubleType())) \
                           .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # Create a VectorAssembler for the distance_km feature
    assembler = VectorAssembler(inputCols=["distance_km"], outputCol="features")
    train_data_with_features = assembler.transform(train_df)

    # Create a Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol="fare_amount")

    # Fit the model
    model = lr.fit(train_data_with_features)

    # Save the model
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Training Complete] Model saved to -> {MODEL_PATH}")
else:
    print(f"[Model Found] Using existing model from {MODEL_PATH}")

# --- PART 2: STREAMING INFERENCE ---
print("\n[Inference Phase] Starting real-time fare prediction stream...")

# Define the schema for incoming JSON data from the socket
schema = StructType([
    StructField("trip_id", StringType()),
    StructField("driver_id", IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp", StringType())
])

# Read the streaming data from the socket
raw_stream = spark.readStream.format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Parse JSON input
parsed_stream = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Load pre-trained model
model = LinearRegressionModel.load(MODEL_PATH)

# Create the same VectorAssembler for inference
assembler_inference = VectorAssembler(inputCols=["distance_km"], outputCol="features")
stream_with_features = assembler_inference.transform(parsed_stream)

# Apply model for prediction
predictions = model.transform(stream_with_features)

# Calculate deviation between actual and predicted fares
predictions_with_deviation = predictions.withColumn("deviation", abs_diff(col("fare_amount") - col("prediction")))

# Select output columns for console display
output_df = predictions_with_deviation.select(
    "trip_id", "driver_id", "distance_km", "fare_amount",
    col("prediction").alias("predicted_fare"), "deviation"
)

# Write output to console
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

# Keep stream running
query.awaitTermination()
