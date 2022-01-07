# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Test registered model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This notebook will load the registered model from the shared repository and test it against test data

# COMMAND ----------

registry_uri="databricks://rmr:rmr"

# COMMAND ----------

#Read the data from the data lake
diabetes = spark.read.format('csv').options(
    header='true', inferschema='true').load("/mnt/modelData/test/diabetes.csv")

display(diabetes)

# COMMAND ----------

# Load the model in the Staging status
import mlflow

model_name = "SparkModel"

client = mlflow.tracking.MlflowClient(registry_uri=registry_uri)
pre_production_model = client.get_latest_versions(name = model_name, stages = ["Staging"])[0]

#print(pre_production_model)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Retrieve the registered model

# COMMAND ----------

import mlflow.pyfunc

mlflow.set_registry_uri(registry_uri)

model_version_uri = f"models:/{pre_production_model.name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")

model_version_1 = mlflow.pyfunc.spark_udf(spark, model_uri=model_version_uri)

# COMMAND ----------

  #Remove the label column from the data. Here we are using the same data we used to train/test the model as it's a demo. In reality you will be loading real test data from your data lake
  X_test=diabetes.drop("Outcome")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Score model with test data

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct
# Predict on a Spark DataFrame.
columns = list(diabetes.columns)
predicted_df = diabetes.withColumn("prediction", model_version_1(struct('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')))
display(predicted_df)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Evaluate the model metrics

# COMMAND ----------

#Calculate Accuracy

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

eval_accuracy = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="accuracy")

accuracy = eval_accuracy.evaluate(predicted_df)

# COMMAND ----------

print(accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC You can also throw an error if the accuracy of the tested model is not above a certain percentage

# COMMAND ----------

#Now check if accuracy>0.7 

#asset accuracy>0.77
