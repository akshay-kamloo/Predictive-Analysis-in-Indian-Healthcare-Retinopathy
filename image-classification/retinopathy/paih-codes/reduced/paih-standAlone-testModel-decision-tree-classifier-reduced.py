from pyspark.ml.classification import OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from sparkdl import readImages
from pyspark.sql import Row
from sparkdl import DeepImageFeaturizer
from pyspark.ml.classification import OneVsRestModel
from pyspark.sql.functions import lit

# Prepare Test Data

imageDir = imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/dst/resized/"

labelZeroDf = readImages(imageDir + "tl0").withColumn("label", lit(0))
labelOneDf = readImages(imageDir + "tl1").withColumn("label", lit(1))
labelTwoDf = readImages(imageDir + "tl2").withColumn("label", lit(2))
labelThreeDf = readImages(imageDir + "tl3").withColumn("label", lit(3))
labelFourDf = readImages(imageDir + "tl4").withColumn("label", lit(4))
finalTestDf = labelZeroDf.unionAll(labelOneDf).unionAll(labelTwoDf).unionAll(labelThreeDf).unionAll(labelFourDf)


testSize = finalTestDf.count()
print(str(testSize))
featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")
featureVector = featurizer.transform(finalTestDf)
model_dc = OneVsRestModel.load(imageDir + 'model-decision-tree-classifier-new')

predictions = model_dc.transform(featureVector)
predictions.show()

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Data set accuracy with decision-tree-classifieer for " + str(testSize) + " images = " + str(accuracy) + " and error " + str(1 - accuracy))


evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="weightedPrecision")
weightedPrecision = evaluator.evaluate(predictions)
print("Test Data set weightedPrecision with decision-tree-classifier for " + str(testSize) + " images = " + str(weightedPrecision))

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="weightedRecall")
weightedRecall = evaluator.evaluate(predictions)
print("Test Data set weighted Recall with decision-tree-classifier for " + str(testSize) + " images = " + str(weightedRecall))

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="f1")
f1Score = evaluator.evaluate(predictions)
print("Train Data set f1 score with decision-tree-classifier for " + str(testSize) + " images = " + str(f1Score))


#predictionAndLabels = predictions.map(lambda lp : (float(lp[
#metrics = MulticlassMetrics(predictionAndLabels)
#precision = metrics.precision()
#recall = metrics.recall()
#f1Score = metrics.fMeasure()

#print("precision = " + str(precision) + " Recall = " + str(recall) + " f1 Score = " + str(f1Score))
