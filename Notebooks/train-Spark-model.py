# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Example of a simple Spark ML model trained on diabetes data

# COMMAND ----------

# MAGIC %md ---
# MAGIC 
# MAGIC ## Data
# MAGIC We will use the diabetes dataset for this experiement, a well-known small dataset.  This cell loads the dataset and splits it into random training and testing sets.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Import the relevant libraries

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

#Read the data from the lake
diabetes = spark.read.format('csv').options(
    header='true', inferschema='true').load("/mnt/modelData/test/diabetes.csv")

display(diabetes)

# COMMAND ----------

#Check for null values
from pyspark.sql.functions import isnan, when, count, col,isnull

nullDF=diabetes.select([count(when(isnull(c), c)).alias(c) for c in diabetes.columns])

display(nullDF)

# COMMAND ----------

#Print the dataframe schema
diabetes.printSchema()

# COMMAND ----------


# Filter for just numeric columns (and exclude Outcome, our label)
numericCols = [field for (field, dataType) in diabetes.dtypes if (((dataType == "int") or(dataType == "double")) & (field != "Outcome"))]
# Combine output of StringIndexer defined above and numeric columns
#assemblerInputs = indexOutputCols + numericCols
assemblerInputs = numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

#Split the data in train and test data
(trainDF, testDF) = diabetes.randomSplit([.8, .2], seed=42)



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Create an initial Random Forest Model

# COMMAND ----------

rfw = RandomForestClassifier(labelCol="Outcome", featuresCol="features", numTrees=10, seed=42)

# Combine stages into pipeline
stages = [vecAssembler, rfw]

pipeline = Pipeline(stages=stages)



# COMMAND ----------

# MAGIC %md ---
# MAGIC ## Train the model

# COMMAND ----------

# Train model with Training Data
pipelineModel = pipeline.fit(trainDF)



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Score the model on test data

# COMMAND ----------

#Scoring the test data

predDF = pipelineModel.transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Print metrics

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator



eval_accuracy = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="accuracy")

accuracy = eval_accuracy.evaluate(predDF)

print(accuracy)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Set the registry_uri to the shared registry

# COMMAND ----------

registry_uri="databricks://rmr:rmr"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Store the experiment metrics in the local registry and the trained model in the shared registry

# COMMAND ----------

# Use ML Flow to register the experiments in the local registry and register the trained model in the shared registry
import mlflow
import mlflow.spark

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


with mlflow.start_run(run_name="SparkML-Test") as run:

  # Log the algorithm parameter num_trees to the run
  mlflow.log_param('num_trees', 10)


  # Train model with Training Data
  pipelineModel = pipeline.fit(trainDF)

  #Scoring the test data
  predDF = pipelineModel.transform(testDF)
  
  #Evaluate the accuracy metrics
  eval_accuracy = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="accuracy")

  accuracy = eval_accuracy.evaluate(predDF)

  print('Accuracy is', str(accuracy))
  mlflow.log_param('Accuracy', accuracy)
  

  # Log model
  print("Stage: Log Model Pipeline - Status: Started")
  mlflow.spark.log_model(pipelineModel, "model")
  print("Stage: Log Model Pipeline - Status: Complete")



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Register the model in the shared registry

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)

# COMMAND ----------

#Register the mode with name: SparkModel
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name="SparkModel")

# COMMAND ----------

#Set the status of the model to "Staging" as it's redy to be tested in the test environment


from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="SparkModel", version=1)
model_version_details.status


# COMMAND ----------

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage="Staging",
)
