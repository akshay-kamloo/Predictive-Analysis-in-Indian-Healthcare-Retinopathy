from sparkdl import readImages
from pyspark.sql import Row
import os.path
from sparkdl import DeepImageFeaturizer
from pyspark.ml.classification import OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit
imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/resized/"

# Prepare Test Data
labelZeroDf = readImages(imageDir + "tl0").withColumn("label", lit(0))
labelOneDf = readImages(imageDir + "tl1").withColumn("label", lit(1))
labelTwoDf = readImages(imageDir + "tl2").withColumn("label", lit(2))
labelThreeDf = readImages(imageDir + "tl3").withColumn("label", lit(3))
labelFourDf = readImages(imageDir + "tl4").withColumn("label", lit(4))
finalTestDf = labelZeroDf.unionAll(labelOneDf).unionAll(labelTwoDf).unionAll(labelThreeDf).unionAll(labelFourDf)
testSize = finalTestDf.count()
featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")
featureVector = featurizer.transform(finalTestDf)
model_lr = OneVsRestModel.load(imageDir + 'model-logistic-regression-new')
predictions = model_lr.transform(featureVector)
predictions.show(predictions.count())
#evaluator = MulticlassClassificationEvaluator(metricName="accuracy")


evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Data set accuracy with Logistic Regression for " + str(testSize) + " images = " + str(accuracy) + " and error " + str(1 - accuracy))


evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="weightedPrecision")
weightedPrecision = evaluator.evaluate(predictions)
print("Test Data set weightedPrecision with Logistic Regression for " + str(testSize) + " images = " + str(weightedPrecision))

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="weightedRecall")
weightedRecall = evaluator.evaluate(predictions)
print("Test Data set weighted Recall with Logistic Regression for " + str(testSize) + " images = " + str(weightedRecall))

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="f1")
f1Score = evaluator.evaluate(predictions)
print("Test Data set f1 with Logistic Regression for " + str(testSize) + " images = " + str(f1Score))
