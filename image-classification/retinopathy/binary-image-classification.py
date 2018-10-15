from sparkdl import readImages
from pyspark.sql.functions import lit

#define data directory
imgDir = "/home/mvk/image-classification/sushant"

#read Images for training data without diabetic-retinopathy as negDf
negDf = readImages(imgDir + "/training/negative").withColumn("label", lit(0))

#read Images for training data with diabetic-retinopathy as posDf  
posDf = readImages(imgDir + "/training/positive").withColumn("label", lit(1))

#read Images for test data without diabetic-retinopathy as testNegDf
testNegDf = readImages(imgDir + "/test/negative").withColumn("label", lit(0))

#read Images for test data with diabetic-retinopathy as testPosDf
testPosDf = readImages(imgDir + "/test/positive").withColumn("label", lit(1))

#prepare complete training data by combining negDf and posDf
trainDf = negDf.unionAll(posDf)

#prepare complete test data by combining testNegDf and testPosDf
testDf = testNegDf.unionAll(testPosDf)

#various required imports
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

#use of DeepImageFeaturizer for extracting features for image classification
#for provided input images
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")

#defining machine learning model for Logistic Regression with parameter initialization
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")

#Instructing Spark to pipeline execution of featurization and Logistic Regression
p = Pipeline(stages=[featurizer, lr])

#Feeding training data to pipeline to create model
pModel = p.fit(trainDf)

#predicting output for test data
predictions = pModel.transform(testDf)

#displaying predictions for test data
predictions.select("filePath", "prediction").show(truncate=False)

#importing MulticlassClassificationEvaluator for accuracy check
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#recording predictions for testdata
df = pModel.transform(testDf)
df.show()

#Printing actual and Predicted output for test data
predictionAndLabels = df.select("prediction", "label")
predictionAndLabels.show()

#calculate and print accuracy and error for test data
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test Data set accuracy = " + str(evaluator.evaluate(predictionAndLabels)) + " and error " + str(1 - evaluator.evaluate(predictionAndLabels)))
