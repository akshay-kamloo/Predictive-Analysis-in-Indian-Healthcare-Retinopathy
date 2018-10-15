from PIL import Image
import os
import sys

if len(sys.argv) < 3 :
	print "Usage: resizeGscale.py <inputDir> <outputDir>"
	sys.exit()

outputDir = sys.argv[2]
inputDir = sys.argv[1]

filesList = os.listdir(inputDir)
#filesList.remove('gscale')

if not os.path.exists(outputDir):
	os.makedirs(outputDir)

img_rows, img_cols = 400, 400
for file in filesList:
	base = os.path.basename(inputDir + "/" + file)
	fileName = os.path.splitext(base)[0]
	im = Image.open(inputDir + "/" + file)
	img = im.resize((img_rows,img_cols)).convert('RGB').save(outputDir + "/"  + fileName, 'jpeg')


