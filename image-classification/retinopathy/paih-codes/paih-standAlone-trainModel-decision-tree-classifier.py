from pyspark.ml.classification import DecisionTreeClassifier, OneVsRest
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
tmpTrainRDD = tmpTrainDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmpTrainX = tmpTrainRDD.toDF()
csvTrainTmp = spark.read.format("csv").option("header", "true").load(imageDir + "train25.csv")
																													csvTrainRDD = csvTrainTmp.rdd.map(lambda x : Row(image = x[0], label = int(x[1])))
																													csvTrain = csvTrainRDD.toDF()
																													finalTrainDataFrame = tmpTrainX.join(csvTrain, tmpTrainX.fileName == csvTrain.image, 'inner').drop(csvTrain.image)

																													featurizer = DeepImageFeaturizer(inputCol="image",outputCol="features", modelName="InceptionV3")

																													method = DecisionTreeClassifier(labelCol="label", featuresCol="features")
																													ovr = OneVsRest(classifier = method)
																													featureVector = featurizer.transform(finalTrainDataFrame).persist()
																													model_dc = ovr.fit(featureVector)
																													model_dc.write().overwrite().save(imageDir + 'model-decision-tree-classifier')

																													predictions = model_dc.transform(featureVector).persist()
																							#predictionAndLabels.persist()
#predictionAndLabels.show()
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Train Data set accuracy with Decision Tree Classifier= " + str(evaluator.evaluate(predictions)) + " and error " + str(1 - evaluator.evaluate(predictions)))


