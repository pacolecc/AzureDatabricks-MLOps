# Databricks notebook source
# MAGIC %md
# MAGIC # This notebook promotes the model from Staging to Production

# COMMAND ----------

#Import all the right libraries

import mlflow

import mlflow.spark



# COMMAND ----------

#Set the remote registry url
registry_uri="databricks://rmr:rmr"

# COMMAND ----------

#Set the remote registry object
mlflow.set_registry_uri(registry_uri)

# COMMAND ----------

#Load the SparkModel(this is the model name)
from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="SparkModel", version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Set the statge of the model to Production

# COMMAND ----------

#Set change the stage of the model to Production
client.transition_model_version_stage(
  name=model_version_details.name,
  version=model_version_details.version,
  stage="Production",
)


# COMMAND ----------

#Print the model stage to console

model_version_details = client.get_model_version(
  name=model_version_details.name,
  version=model_version_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")
