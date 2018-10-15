from sparkdl import readImages
from pyspark.sql.functions import lit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
import org.apache.spark.sql
import org.apache.spark.sql._


img_dir = "/home/mvk/images_classification-master/personalities"

jobs_df = readImages(img_dir + "/jobs").withColumn("label", lit(1))
zuckerberg_df = readImages(img_dir + "/zuckerberg").withColumn("label", lit(0))
jobs_train, jobs_test = jobs_df.randomSplit([0.6, 0.4])
zuckerberg_train, zuckerberg_test = zuckerberg_df.randomSplit([0.6, 0.4])

train_df = jobs_train.unionAll(zuckerberg_train)
test_df = jobs_test.unionAll(zuckerberg_test)

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)
predictions = p_model.transform(test_df)

predictions.select("filePath", "prediction").show(truncate=False)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
df = p_model.transform(test_df)
df.show()

predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
