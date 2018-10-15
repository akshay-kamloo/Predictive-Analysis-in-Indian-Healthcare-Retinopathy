from pyspark.ml.classification import OneVsRest
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
tmpTestDf = readImages(imageDir + "test25")
tmpTestRDD = tmpTestDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmptestX = tmpTestRDD.toDF()
csvTestTmp = spark.read.format("csv").option("header", "true").load(imageDir + "test25.csv")
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
#print gc.collect()

#model_nb = OneVsRestModel.load('hdfs://192.168.65.188:8020/paih/model-naive-bayes')
model_dc = OneVsRestModel.load(imageDir + 'model-decision-tree-classifier')

predictions = model_dc.transform(featureVector)
predictions.show()

print '***Transform complete***'
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test Data set accuracy with decision tree classifier for 25 images = " + str(evaluator.evaluate(predictions)) + " and error " + str(1 - evaluator.evaluate(predictions)))
