from sparkdl import readImages
from pyspark.sql import Row
import os.path
from sparkdl import DeepImageFeaturizer
from pyspark.ml.classification import OneVsRestModel
#from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#sc = SparkContext()
#spark = SparkSession(sc)

imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/"
#imageDir = "hdfs://192.168.65.188:8020/paih/"

def getFileName (filePath) :
	fileName = os.path.basename(filePath).split(".")[0]
	return fileName

# Prepare Test Data
tmpTestDf = readImages(imageDir + "test5")
tmpTestRDD = tmpTestDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmptestX = tmpTestRDD.toDF()
csvTestTmp = spark.read.format("csv").option("header", "true").load(imageDir + "test5.csv")
csvTestRDD = csvTestTmp.rdd.map(lambda x : Row(image = x[0], label = int(x[1])))
csvTest = csvTestRDD.toDF()
finalTestDataFrame = tmptestX.join(csvTest, tmptestX.fileName == csvTest.image, 'inner').drop(csvTest.image)

featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")
featureVector = featurizer.transform(finalTestDataFrame)
model_dr = OneVsRestModel.load(imageDir + 'model-dicision-tree-regression')
predictions = model_dr.transform(featureVector)
predictions.persist()
predictions.select("filePath", "prediction").show(truncate=False)

predictionAndLabels = predictions.select("prediction", "label")

#predictionAndLabels.show()

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test Data set accuracy with Decision tree regression = " + str(evaluator.evaluate(predictionAndLabels)) + " and error " + str(1 - evaluator.evaluate(predictionAndLabels)))

#regression evaluator
#evaluator = RegressionEvaluator(
 #   labelCol="label", predictionCol="prediction", metricName="rmse")
#rmse = evaluator.evaluate(predictions)
#print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
#predictions.show()
