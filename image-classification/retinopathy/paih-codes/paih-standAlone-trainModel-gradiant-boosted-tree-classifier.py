from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkdl import readImages
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from sparkdl import DeepImageFeaturizer
import os.path

#sc = SparkContext()
#spark = SparkSession(sc)

imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/"
#imageDir = "hdfs://192.168.65.188:8020/paih/"

def getFileName (filePath) :
	fileName = os.path.basename(filePath).split(".")[0]
	return fileName

# Prepare Train Data
tmpTrainDf = readImages(imageDir + "train25")
#tmpTrainDf = readImages(imageDir + "/test1")
tmpTrainRDD = tmpTrainDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmpTrainX = tmpTrainRDD.toDF()
csvTrainTmp = spark.read.format("csv").option("header", "true").load(imageDir + "train25.csv")
#csvTrainTmp = spark.read.format("csv").option("header", "true").load(imageDir + "/test1.csv")
csvTrainRDD = csvTrainTmp.rdd.map(lambda x : Row(image = x[0], label = int(x[1])))
csvTrain = csvTrainRDD.toDF()
finalTrainDataFrame = tmpTrainX.join(csvTrain, tmpTrainX.fileName == csvTrain.image, 'inner').drop(csvTrain.image)

featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")

method = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
ovr = OneVsRest(classifier = method)
featureVector = featurizer.transform(finalTrainDataFrame).persist()
model_gbt = ovr.fit(featureVector)
model_gbt.write().overwrite().save(imageDir + 'model-gradiant-boosted-tree-classifier')

predictions = model_gbt.transform(featureVector).persist()

predictionAndLabels = predictions.select("prediction", "label")
predictionAndLabels.persist()
#predictionAndLabels.show()
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Train Data set Gradiant boosted tree classifier = " + str(evaluator.evaluate(predictionAndLabels)) + " and error " + str(1 - evaluator.evaluate(predictionAndLabels)))


