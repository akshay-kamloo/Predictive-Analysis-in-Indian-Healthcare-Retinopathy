from pyspark.ml.classification import DecisionTreeClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkdl import readImages
from pyspark.sql import Row
from sparkdl import DeepImageFeaturizer
import os.path
from pyspark.sql.functions import lit


imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/dst/resized/"

# Prepare Train Data

labelZeroDf = readImages(imageDir + "l0").withColumn("label", lit(0))
labelOneDf = readImages(imageDir + "l1").withColumn("label", lit(1))
labelTwoDf = readImages(imageDir + "l2").withColumn("label", lit(2))
labelThreeDf = readImages(imageDir + "l3").withColumn("label", lit(3))
labelFourDf = readImages(imageDir + "l4").withColumn("label", lit(4))
finalTrainDf = labelZeroDf.unionAll(labelOneDf).unionAll(labelTwoDf).unionAll(labelThreeDf).unionAll(labelFourDf)

trainSize = finalTrainDf.count()
print(str(trainSize))

featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")
method = DecisionTreeClassifier(labelCol="label",featuresCol="features")                                  
ovr = OneVsRest(classifier = method)
featureVector = featurizer.transform(finalTrainDf)
model_dc = ovr.fit(featureVector)
model_dc.write().overwrite().save(imageDir + 'model-decision-tree-classifier-new')
predictions = model_dc.transform(featureVector)

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Train Data set accuracy with decision-tree-classifier for " + str(trainSize) + " images = " + str(accuracy) + " and error " + str(1 - accuracy))



