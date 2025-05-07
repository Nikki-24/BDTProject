# Databricks notebook source
pip install pymongo


# COMMAND ----------

pip install pandas matplotlib seaborn


# COMMAND ----------


import requests
import json

url = "https://data.ct.gov/api/views/5mzw-sjtu/rows.json?accessType=DOWNLOAD"  

response = requests.get(url)
response.raise_for_status() 
data = response.json()

# COMMAND ----------

c=data.get('meta').get('view').get('columns')
col=[col['name'] for col in c]

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(data.get('data', []), columns=col)

# COMMAND ----------

df.head()

# COMMAND ----------

df=df.iloc[:,8:]
df.head()

# COMMAND ----------

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://nikki:nikki123@cluster0.hqc9i.mongodb.net/?retryWrites=true&w=majority&connectTimeoutMS=300000"
client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# COMMAND ----------

import pymongo
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import AutoReconnect

db = client['BigDataProject']
collection = db['RealEstateData']

data = df.to_dict(orient='records')

try:
    batch_size = 10000
    for i in range(0, 120000, batch_size):
        batch = data[i : i + batch_size]
        collection.insert_many(batch, ordered=False)
except AutoReconnect as e:
    print(f"AutoReconnect error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()  

# COMMAND ----------

uri = "mongodb+srv://nikki:nikki123@cluster0.hqc9i.mongodb.net/?retryWrites=true&w=majority&connectTimeoutMS=300000"
client = MongoClient(uri, server_api=ServerApi('1'))

# COMMAND ----------

db = client['BigDataProject']
collection = db['RealEstateData']
RealEstateData = pd.DataFrame(list(collection.find()))

# COMMAND ----------

RealEstateData.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### DATA PROFILING 

# COMMAND ----------

RealEstateData.columns

# COMMAND ----------

columns_to_drop = ["_id", "Town Index", "Planning Regions"]
RealEstateData = RealEstateData.drop(columns=columns_to_drop)

# Show the resulting data
print(RealEstateData.head())
print(RealEstateData.info())



# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("RealEstate").getOrCreate()

# Convert Pandas DataFrame to PySpark DataFrame
sdf = spark.createDataFrame(RealEstateData)
sdf.show()




# COMMAND ----------

sdf.describe().show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Find nulls 

# COMMAND ----------

# Count nulls in each column
from pyspark.sql.functions import col, isnan, when, count

sdf.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in sdf.columns
]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### cleaning data by null count

# COMMAND ----------

sdf = sdf.dropna(subset=["Serial Number", "Date Recorded","Assessed Value", "Sale Amount", "Address", "Town"])
columns_to_drop = ["OPM remarks", "Assessor Remarks"]
sdf_cleaned = sdf.drop(*columns_to_drop)


# COMMAND ----------

sdf_cleaned.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Manipulation  Silver layer

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType


sdf_cleaned = sdf_cleaned.withColumn("Assessed Value", sdf_cleaned["Assessed Value"].cast(DoubleType()))
sdf_cleaned = sdf_cleaned.withColumn("Sale Amount", sdf_cleaned["Sale Amount"].cast(DoubleType()))
sdf_cleaned = sdf_cleaned.withColumn("Sales Ratio", sdf_cleaned["Sales Ratio"].cast(DoubleType()))
sdf_cleaned = sdf_cleaned.withColumn("List Year", sdf_cleaned["List Year"].cast(IntegerType()))
sdf_cleaned = sdf_cleaned.withColumn("Date Recorded", F.to_timestamp(sdf_cleaned["Date Recorded"], "yyyy-MM-dd'T'HH:mm:ss"))
sdf_cleaned = sdf_cleaned.filter((sdf_cleaned["Sales Ratio"] >= 0) & (sdf_cleaned["Sales Ratio"] <= 1))
sdf_cleaned = sdf_cleaned.withColumn("Address", F.trim(sdf_cleaned["Address"]))
sdf_cleaned = sdf_cleaned.withColumn("Location", F.trim(sdf_cleaned["Location"]))
sdf_cleaned = sdf_cleaned.dropna(subset=["Zip Code", "Counties"])
sdf_cleaned = sdf_cleaned.drop("Non Use Code")

sdf_cleaned.printSchema()

sdf_cleaned.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregated_datasets Gold Layer

# COMMAND ----------

from pyspark.sql.functions import col, year, month, avg, sum, count, when


# COMMAND ----------

# MAGIC %md
# MAGIC ####  Top 10 Zip Codes by Total Sale Amount

# COMMAND ----------

sdf_cleaned.groupBy("Zip Code") \
  .agg(sum("Sale Amount").alias("Total_Sales")) \
  .orderBy(col("Total_Sales").desc()) \
  .limit(10) \
  .show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Average Sales Ratio by Town

# COMMAND ----------

sdf_cleaned.groupBy("Town").agg(avg("Sales Ratio").alias("Avg_Sales_Ratio")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Average Assessed Value by Property Type

# COMMAND ----------

sdf_cleaned.groupBy("Property Type") \
  .agg(avg("Assessed Value").alias("Avg_Assessed_Value")) \
  .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Distribution of Sales Ratio (binned)

# COMMAND ----------

sdf_cleaned.withColumn("Sales_Bin", when(col("Sales Ratio") < 0.2, "0-0.2")
                            .when(col("Sales Ratio") < 0.4, "0.2-0.4")
                            .when(col("Sales Ratio") < 0.6, "0.4-0.6")
                            .when(col("Sales Ratio") < 0.8, "0.6-0.8")
                            .otherwise("0.8+")) \
  .groupBy("Sales_Bin") \
  .count() \
  .orderBy("Sales_Bin") \
  .show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Count of Properties by Residential Type

# COMMAND ----------


sdf_cleaned.groupBy("Residential Type").count().show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Monthly Trends in Property Sales

# COMMAND ----------

sdf_cleaned.withColumn("Month", month("Date Recorded")) \
  .groupBy("Month") \
  .agg(count("*").alias("Monthly_Sales_Count")) \
  .orderBy("Month") \
  .show()


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# features
indexers = [
    StringIndexer(inputCol="Town", outputCol="Town_index"),
    StringIndexer(inputCol="Property Type", outputCol="Property_index"),
    StringIndexer(inputCol="Residential Type", outputCol="Residential_index")
]

assembler = VectorAssembler(
    inputCols=["Assessed Value", "Town_index", "Property_index", "Residential_index"],
    outputCol="features"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression – Predict Sale Amount

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

df_reg = sdf_cleaned.select("Sale Amount", "Assessed Value", "Town", "Property Type", "Residential Type").na.drop()

lr = LinearRegression(featuresCol="features", labelCol="Sale Amount")
pipeline = Pipeline(stages=indexers + [assembler, lr])

model = pipeline.fit(df_reg)
predictions = model.transform(df_reg)
predictions.select("Sale Amount", "prediction").show(5)


# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="Sale Amount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(
    labelCol="Sale Amount", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

evaluator_mae = RegressionEvaluator(
    labelCol="Sale Amount", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)

print(f"Linear Regression RMSE: {rmse}")
print(f"Linear Regression R²: {r2}")
print(f"Linear Regression MAE: {mae}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Clustering – KMeans on sale-related features

# COMMAND ----------

from pyspark.ml.clustering import KMeans

df_cluster = sdf_cleaned.select("Sale Amount", "Assessed Value", "Sales Ratio").na.drop()
vec_assembler = VectorAssembler(inputCols=["Sale Amount", "Assessed Value", "Sales Ratio"], outputCol="features")
df_features = vec_assembler.transform(df_cluster)

kmeans = KMeans(k=4, seed=1)
model = kmeans.fit(df_features)
clusters = model.transform(df_features)
clusters.select("Sale Amount", "Assessed Value", "Sales Ratio", "prediction").show(5)



# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(clusters)

print(f"KMeans Silhouette Score: {silhouette}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization

# COMMAND ----------

top_towns_df = sdf_cleaned.groupBy("Town").agg(sum("Sale Amount").alias("Total_Sales")) \
                 .orderBy(col("Total_Sales").desc()).limit(10)

top_towns_pd = top_towns_df.toPandas()

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x="Total_Sales", y="Town", data=top_towns_pd)
plt.title("Top 10 Towns by Sale Amount")
plt.show()


# COMMAND ----------

monthly_df = sdf_cleaned.withColumn("Month", month("Date Recorded")) \
    .groupBy("Month") \
    .agg(count("*").alias("Sales_Count")) \
    .orderBy("Month")

monthly_pd = monthly_df.toPandas()

sns.lineplot(data=monthly_pd, x="Month", y="Sales_Count", marker="o")
plt.title("Monthly Property Sales Count")
plt.xlabel("Month")
plt.ylabel("Number of Sales")
plt.xticks(range(1,13))
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------

scatter_df = sdf_cleaned.select("Assessed Value", "Sale Amount").dropna().toPandas()

sns.scatterplot(data=scatter_df, x="Assessed Value", y="Sale Amount", alpha=0.5)
plt.title("Assessed Value vs Sale Amount")
plt.xlabel("Assessed Value")
plt.ylabel("Sale Amount")
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------

top_towns = sdf_cleaned.groupBy("Town").count().orderBy(col("count").desc()).limit(5)
top_town_list = [row["Town"] for row in top_towns.collect()]

filtered_df = sdf_cleaned.filter(col("Town").isin(top_town_list)).select("Town", "Sales Ratio").dropna()
filtered_pd = filtered_df.toPandas()

sns.boxplot(data=filtered_pd, x="Town", y="Sales Ratio")
plt.title("Sales Ratio Distribution by Top 5 Towns")
plt.ylabel("Sales Ratio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

res_type_df = sdf_cleaned.groupBy("Residential Type").count().orderBy("count", ascending=False)
res_type_pd = res_type_df.toPandas()

plt.figure(figsize=(8, 8))
plt.pie(
    res_type_pd["count"],
    labels=res_type_pd["Residential Type"],
    autopct="%1.1f%%",
    startangle=140,
    shadow=True
)
plt.title("Distribution of Properties by Residential Type")
plt.axis("equal")  
plt.tight_layout()
plt.legend()
plt.show()


# COMMAND ----------

