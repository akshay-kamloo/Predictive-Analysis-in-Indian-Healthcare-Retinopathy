from sparkdl import readImages
from pyspark.sql import Row

imageDir = "/media/prateek/2A8BEA421AC55874/PAIH/work"
#imageDir = "hdfs://192.168.65.188:8020/paih/dataset"

def getFileName (filePath) :
	fileName = os.path.basename(filePath).split(".")[0]
	return fileName

# Prepare Train Data
tmpTrainDf = readImages(imageDir + "/train25")
tmpTrainRDD = tmpTrainDf.rdd.map(lambda x : Row(filepath = x[0], image = x[1], fileName = getFileName(x[0])))
tmpTrainX = tmpTrainRDD.toDF()

