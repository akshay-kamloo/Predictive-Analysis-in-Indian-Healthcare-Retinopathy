from pyspark.sql.functions import lit
from pyspark.ml.classification import OneVsRest, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkdl import readImages
from sparkdl import DeepImageFeaturizer

imageDir = imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/resized/"

labelZeroDf = readImages(imageDir + "l0").withColumn("label", lit(0))
labelOneDf = readImages(imageDir + "l1").withColumn("label", lit(1))
labelTwoDf = readImages(imageDir + "l2").withColumn("label", lit(2))
labelThreeDf = readImages(imageDir + "l3").withColumn("label", lit(3))
labelFourDf = readImages(imageDir + "l4").withColumn("label", lit(4))
finalTrainDf = labelZeroDf.unionAll(labelOneDf).unionAll(labelTwoDf).unionAll(labelThreeDf).unionAll(labelFourDf)
trainSize = finalTrainDf.count()
featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")

method = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="label")
ovr = OneVsRest(classifier = method)
featureVector = featurizer.transform(finalTrainDf)
model_nb = ovr.fit(featureVector)
model_nb.write().overwrite().save(imageDir + 'model-naive-bayes-new')

predictions = model_nb.transform(featureVector)
predictions.show(predictions.count())

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Train Data set accuracy with Naive Bayes for " + str(trainSize) + " images = " + str(accuracy) + " and error " + str(1 - accuracy))




