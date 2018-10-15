from pyspark.sql.window import Window
form pyspark.sql.functions import row_number
from sparkdl import readImages
csv = spark.read.format("csv").option("header", "true").load("PATH/TO/CSV").select('level').limit(100).withColumn("rowNumber"), row_number().over(Window.partitionBy("level").orderBy("level")))
retinopathyDf = readImages("PATH/TO/IMAGES").limit(100).withColumn("rowNumber", row_number().over(Window.partitionBy("filepath").orderBy("image")))
fianlDataFrame = retinopathyDf.join(csv, retinopathyDf.rowNumber == csv.rowNumber, 'inner').drop(csv.rowNumber)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.import Pipeline
from sparkdl import DeepImageFeaturizer
featurizer = DeepImagerFeaturizer(inputCol = "image", outputCol="features", modelName="inceptionV3")
lr = LogsticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="level")
p = Pipeline(stages=[featurizer, lr])

