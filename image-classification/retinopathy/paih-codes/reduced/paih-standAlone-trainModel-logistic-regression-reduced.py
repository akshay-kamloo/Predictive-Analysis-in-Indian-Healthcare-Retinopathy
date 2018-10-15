from sparkdl import readImages
from pyspark.sql import Row
import os.path
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.classification import OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkdl import DeepImageFeaturizer
from pyspark.sql.functions import lit

#sc = SparkContext()
#spark = SparkSession(sc)

imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work/resized/"

# Prepare Train Data
labelZeroDf = readImages(imageDir + "l0").withColumn("label", lit(0))
labelOneDf = readImages(imageDir + "l1").withColumn("label", lit(1))
labelTwoDf = readImages(imageDir + "l2").withColumn("label", lit(2))
labelThreeDf = readImages(imageDir + "l3").withColumn("label", lit(3))
labelFourDf = readImages(imageDir + "l4").withColumn("label", lit(4))
finalTrainDf = labelZeroDf.unionAll(labelOneDf).unionAll(labelTwoDf).unionAll(labelThreeDf).unionAll(labelFourDf)
trainSize = finalTrainDf.count()


featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")

method = LogisticRegression(maxIter=50, regParam=0.05, elasticNetParam=0.3, labelCol="label")
ovr = OneVsRest(classifier = method)
featureVector = featurizer.transform(finalTrainDf)
model_lr = ovr.fit(featureVector)
model_lr.write().overwrite().save(imageDir + 'model-logistic-regression-new')
print("Taining completed with " + str(trainSize) + " images.")
predictions = pModel.transform(finalTestDataFrame)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Data set accuracy = " + str(accuracy) + " and error " + str(1 - accuracy))
#predictions.show()

#apply evaluator on constructed model for training data.

#apply deep learning for feature extraction
#apply genetic algo for feature selection
#apply ml model(eg. lr)
#apply genetic algorithm for feature selection


#A. evaluate training and test data by lrand other 10 methods.
#B. Tensoflow and spark cocktail


#p = Pipeline(stages=[featurizer,ovr])
#feature = p.fit(finalTrainDataFrame)
#featureObj.show()
#pModel = p.fit(finalTrainDataFrame)
#p = Pipeline(stages=[ovr])
#pModel = p.fit(finalTrainDataFrame)

#predictions = pModel.transform(finalTestDataFrame)

#predictions.select("filePath", "prediction").show(truncate=False)
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#predictionAndLabels = predictions.select("prediction", "label")
#predictionAndLabels.show()

#evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
#print("Test Data set accuracy = " + str(evaluator.evaluate(predictionAndLabels)) + " and error " + str(1 - evaluator.evaluate(predictionAndLabels)))
