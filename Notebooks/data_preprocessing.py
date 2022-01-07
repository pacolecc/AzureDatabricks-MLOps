# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Read the Data from the Azure Data Lake

# COMMAND ----------

#Read the data from the lake
diabetes = spark.read.format('csv').options(
    header='true', inferschema='true').load("/mnt/modelData/test/diabetes.csv")

display(diabetes)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # Prepare the data as required by the use case

# COMMAND ----------

#To do

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Store the prepared data into the data lake

# COMMAND ----------

#Store prepared data in a chosen location(o.e: delta, Azure Data Lake)
#Below is just an example of storing the prepared data as csv file in the Azure Data Lake leveraging the mounting point /mnt/data
#diabetes.coalesce(1).write.format('csv').options(header='true', inferSchema='true').save('/mnt/modelData/test/processed')


