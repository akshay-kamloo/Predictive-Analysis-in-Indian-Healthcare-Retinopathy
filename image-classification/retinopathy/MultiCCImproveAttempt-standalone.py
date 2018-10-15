from sparkdl import readImages
from pyspark.sql import Row
imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work"

def getFileName (filePath) :
	fileName = os.path.basename(filePath).split(".")[0]
	return fileName

# Prepare Train Data
tmpTrainDf = readImages(imageDir + "/train25")
tmpTrainRDD = tmpTrainDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmpTrainX = tmpTrainRDD.toDF()
csvTrainTmp = spark.read.format("csv").option("header", "true").load(imageDir + "/train25.csv")
csvTrainRDD = csvTrainTmp.rdd.map(lambda x : Row(image = x[0], label = int(x[1])))
csvTrain = csvTrainRDD.toDF()
finalTrainDataFrame = tmpTrainX.join(csvTrain, tmpTrainX.fileName == csvTrain.image, 'inner').drop(csvTrain.image)

# Prepare Test Data
tmpTestDf = readImages(imageDir + "/test5")
tmpTestRDD = tmpTestDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmptestX = tmpTestRDD.toDF()
csvTestTmp = spark.read.format("csv").option("header", "true").load(imageDir + "/test5.csv")
csvTestRDD = csvTestTmp.rdd.map(lambda x : Row(image = x[0], label = int(x[1])))
csvTest = csvTestRDD.toDF()
finalTestDataFrame = tmptestX.join(csvTest, tmptestX.fileName == csvTest.image, 'inner').drop(csvTest.image)

from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")

lr = LogisticRegression(maxIter=50, regParam=0.05, elasticNetParam=0.3, labelCol="label")
ovr = OneVsRest(classifier = lr)

p = Pipeline(stages=[featurizer, ovr])
pModel = p.fit(finalTrainDataFrame)

predictions = pModel.transform(finalTestDataFrame)

predictions.select("filePath", "prediction").show(truncate=False)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictionAndLabels = predictions.select("prediction", "label")
predictionAndLabels.show()

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test Data set accuracy = " + str(evaluator.evaluate(predictionAndLabels)) + " and error " + str(1 - evaluator.evaluate(predictionAndLabels)))





