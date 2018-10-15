from pyspark.sql.functions import lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from sparkdl import readImages
from sparkdl import DeepImageFeaturizer
from pyspark.ml.classification import OneVsRestModel


imageDir = imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/resized/"

labelZeroDf = readImages(imageDir + "tl0").withColumn("label", lit(0))
labelOneDf = readImages(imageDir + "tl1").withColumn("label", lit(1))
labelTwoDf = readImages(imageDir + "tl2").withColumn("label", lit(2))
labelThreeDf = readImages(imageDir + "tl3").withColumn("label", lit(3))
labelFourDf = readImages(imageDir + "tl4").withColumn("label", lit(4))
finalTestDf = labelZeroDf.unionAll(labelOneDf).unionAll(labelTwoDf).unionAll(labelThreeDf).unionAll(labelFourDf)
testSize = finalTestDf.count()
featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")
featureVector = featurizer.transform(finalTestDf)

model_nb = OneVsRestModel.load(imageDir + 'model-naive-bayes-new')
predictions = model_nb.transform(featureVector)
predictions.show(predictions.count())

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Data set accuracy with Naive Bayes for " + str(testSize) + "images = " + str(accuracy) + " and error " + str(1 - accuracy))


evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="weightedPrecision")
weightedPrecision = evaluator.evaluate(predictions)
print("Test Data set weightedPrecision with Naive Bayes  for " + str(testSize) + " images = " + str(weightedPrecision))

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="weightedRecall")
weightedRecall = evaluator.evaluate(predictions)
print("Test Data set weightedPrecision with Naive Bayes  for " + str(testSize) + " images = " + str(weightedRecall))

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="f1")
weightedRecall = evaluator.evaluate(predictions)
print("Train Data set f1 with Naive Bayes for " + str(testSize) + " images = " + str(weightedRecall))


#metrics = MulticlassMetrics(predictions)
#precision = metrics.precision()
#recall = metrics.recall()
#f1Score = metrics.fMeasure()

#print("precision = " + str(precision) + " Recall = " + str(recall) + " f1 Score = " + str(f1Score))



