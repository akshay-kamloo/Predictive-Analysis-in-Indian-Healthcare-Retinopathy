from sparkdl import readImages
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from sparkdl import DeepImageFeaturizer
from pyspark.ml.classification import OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit

sc = SparkContext()
spark = SparkSession(sc)

imageDir = "hdfs://192.168.65.188:8020/paih/rgbGscRgbReducedDataset/"

# Prepare Test Data
labelZeroDf = readImages(imageDir + "tst0").withColumn("label", lit(0))
labelOneDf = readImages(imageDir + "tst1").withColumn("label", lit(1))
labelTwoDf = readImages(imageDir + "tst2").withColumn("label", lit(2))
labelThreeDf = readImages(imageDir + "tst3").withColumn("label", lit(3))
labelFourDf = readImages(imageDir + "tst4").withColumn("label", lit(4))
finalTestDf = labelZeroDf.unionAll(labelOneDf).unionAll(labelTwoDf).unionAll(labelThreeDf).unionAll(labelFourDf)
testSize = finalTestDf.count()
featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")
featureVector = featurizer.transform(finalTestDf)
model_lr = OneVsRestModel.load(imageDir + 'model-logistic-regression-reduced')
predictions = model_lr.transform(featureVector)
predictions.show(testSize)

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Data set accuracy with Logistic Regression for " + str(testSize) + " images = " + str(accuracy) + " and error " + str(1 - accuracy))
