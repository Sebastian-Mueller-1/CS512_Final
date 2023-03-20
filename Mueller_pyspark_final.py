import pyspark
from pyspark.sql import SparkSession
import pprint
import json
from pyspark.sql.types import StructType, FloatType, LongType, StringType, StructField
from pyspark.sql import Window
from math import radians, cos, sin, asin, sqrt
from pyspark.sql.functions import lead, udf, struct, col
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from matplotlib import pyplot
import pandas as pd 

def classify(mag):
  """Give a magnitude a categorical rating based off of magnitude"""
  if mag <1 and mag >0.0:
    return "A"
  if mag <2.0 and mag >= 1.0:
    return "B"
  if mag <3.0 and mag >=2.0:
    return "C"
  if mag <4.0 and mag >=3.0:
    return "D"
  if mag <5.0 and mag >=4.0:
    return "E"
  if mag <6.0 and mag >=5.0:
    return "F"
  if mag <7.0 and mag >=6.0:
    return "G"
  if mag <8.0 and mag >=7.0:  
    return "H"
  if mag <9.0 and mag >=8.0:
    return "I"
  if mag <10 and mag >=9:
    return "J"

def To_numb(x):
  x['Magnitude'] = float(x['Magnitude'])
  x['Location'] = str(x['Location'])
  x['Time'] = int(x['Time'])
  x['Felt'] = int(x['Felt'])
  x['cdi'] = float(x['cdi'])
  x['mmi'] = float(x['mmi'])
  x['Alert'] = str(x['Alert'])
  x['Tsunami'] = int(x['Tsunami'])
  x['sig'] = int(x['sig'])
  x['nst'] = int(x['nst'])
  x['dmin'] = float(x['dmin'])
  x['gap'] = float(x['gap'])
  x['rms'] = float(x['rms'])
  x['Longitude'] = float(x['Longitude'])
  x['Latitude'] = float(x['Latitude'])
  return x
sc = pyspark.SparkContext()

#PACKAGE_EXTENSIONS= ('gs://hadoop-lib/bigquery/bigquery-connector-hadoop2-latest.jar')

bucket = sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = sc._jsc.hadoopConfiguration().get('fs.gs.project.id')
input_directory = 'gs://{}/hadoop/tmp/bigquerry/pyspark_input'.format(bucket)
output_directory = 'gs://{}/pyspark_demo_output'.format(bucket)

spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('Earthquakes') \
  .getOrCreate()

conf={
    'mapred.bq.project.id':project,
    'mapred.bq.gcs.bucket':bucket,
    'mapred.bq.temp.gcs.path':input_directory,
    'mapred.bq.input.project.id': "bigqueryhw",
    'mapred.bq.input.dataset.id': 'eq_data',
    'mapred.bq.input.table.id': "earthquakes_20230319_190319",
}

## pull table from big query
table_data = sc.newAPIHadoopRDD(
    'com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat',
    'org.apache.hadoop.io.LongWritable',
    'com.google.gson.JsonObject',
    conf = conf)

## convert table to a json like object, and use to num to reconvert type
vals = table_data.values()
vals = vals.map(lambda line: json.loads(line))
vals = vals.map(To_numb)

##schema 
schema = StructType([
   StructField('Magnitude', FloatType(), True),
   StructField("Location", StringType(), True),
   StructField("Time", LongType(), True),
   StructField("Felt", LongType(), True),
   StructField("cdi", FloatType(), True),
   StructField('mmi', FloatType(), True),
   StructField("Alert", StringType(), True),
   StructField("Tsunami", LongType(), True),
   StructField("sig", LongType(), True),
   StructField("nst", LongType(), True),
   StructField('dmin', FloatType(), True),
   StructField("rms", FloatType(), True),
   StructField("gap", FloatType(), True),
   StructField("Longitude", FloatType(), True),
   StructField("Latitude", FloatType(), True)])

## create a dataframe object
df1 = spark.createDataFrame(vals, schema= schema)
df1 = df1.dropDuplicates() #drop repeated rows where every column the same

# repartition
df1.repartition(6) 
#-------------------------------ML linear regression-------------------------------
# create vector list of features
feature_vectors = VectorAssembler(inputCols = ['Longitude', 'Latitude'], outputCol = "explanatory variables")
output = feature_vectors.transform(df1)

#collect response variable 
data_for_regression = output.select('explanatory variables', 'Magnitude')
train, test = data_for_regression.randomSplit([0.75,0.25]) #make test train set

# run model 
regressor = LinearRegression(featuresCol = 'explanatory variables', labelCol = 'Magnitude')
regressor = regressor.fit(train)

prediction_results = regressor.evaluate(test)
pprint.pprint(prediction_results.predictions.show())

mag_hist = df1.select('Magnitude').rdd.flatMap(lambda x:x).histogram(11)

#-------------------Preform Classification---------------
classify_udf = udf(classify, StringType())
df1 =df1.withColumn("Class", classify_udf('Magnitude'))

df1.createOrReplaceTempView('Classed')
classes = spark.sql("Select Class, COUNT(*) FROM Classed Group by Class") # ADDED CONDITION FOR LESS THAN 600 MPH
classes = classes.rdd.map(tuple)
pprint.pprint(classes.collect())


input_path = sc._jvm.org.apache.hadoop.fs.Path(input_directory)
input_path.getFileSystem(sc._jsc.hadoopConfiguration()).delete(input_path, True)

