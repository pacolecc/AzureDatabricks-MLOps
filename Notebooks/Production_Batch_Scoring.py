# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #This notebook performs the production batch scoring of a given data set in input

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Read the data to batch score from the data lake but you can also pass an input parameter to this notebook with the path of the data to batch score in the data lake

# COMMAND ----------

#Read the data to batch score from the data lake
diabetes = spark.read.format('csv').options(
    header='true', inferschema='true').load("/mnt/modelData/test/HoldoutDiabetes.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Set the shared registry variable

# COMMAND ----------


registry_uri="databricks://rmr:rmr"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load the SparkModel(this is the model name) which is in Production

# COMMAND ----------

# Retrieve the model in the Production status
import mlflow

model_name = "SparkModel"

client = mlflow.tracking.MlflowClient(registry_uri=registry_uri)
pre_production_model = client.get_latest_versions(name = model_name, stages = ["Production"])[0]

print(pre_production_model)

# COMMAND ----------

#Load the production model
import mlflow.pyfunc

mlflow.set_registry_uri(registry_uri)

model_version_uri = f"models:/{pre_production_model.name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")

model_version_1 = mlflow.pyfunc.spark_udf(spark, model_uri=model_version_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Drop the label from the dataset to score

# COMMAND ----------

 X_test=diabetes.drop("Outcome")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Score the model

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct
# Predict on a Spark DataFrame.
columns = list(diabetes.columns)
predicted_df = X_test.withColumn("prediction", model_version_1(struct('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Display model score dataframe

# COMMAND ----------

display(predicted_df)
