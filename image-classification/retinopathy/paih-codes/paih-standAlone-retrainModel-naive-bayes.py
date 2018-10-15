from pyspark.ml.classification import  OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkdl import readImages
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from sparkdl import DeepImageFeaturizer
from pyspark.ml.classification import OneVsRestModel
import os.path
import gc

#sc = SparkContext()
#spark = SparkSession(sc)

imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/"
#imageDir = "hdfs://192.168.65.188:8020/paih/"

def getFileName (filePath) :
	fileName = os.path.basename(filePath).split(".")[0]
	return fileName

# Prepare Test Data
# Prepare Test Data
tmpTestDf = readImages(imageDir + "train25_2")
tmpTestRDD = tmpTestDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmptestX = tmpTestRDD.toDF()
csvTestTmp = spark.read.format("csv").option("header", "true").load(imageDir + "train25_2.csv")
csvTestRDD = csvTestTmp.rdd.map(lambda x : Row(image = x[0], label = int(x[1])))
csvTest = csvTestRDD.toDF()
finalTestDataFrame = tmptestX.join(csvTest, tmptestX.fileName == csvTest.image, 'inner').drop(csvTest.image)

featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")
featureVector = featurizer.transform(finalTestDataFrame)
del tmpTestDf
del tmpTestRDD
del tmpTestX
del csvTestTmp
del csvTestRDD
del csvTest
del finalTestDataFrame

model_nb = OneVsRestModel.load(imageDir + 'model-naive-bayes')

model_nb.fit(featureVector)
model_nb.write().overwrite().save(imageDir + 'model-naive-bayes-retrained')
print '***re-train complete***'




