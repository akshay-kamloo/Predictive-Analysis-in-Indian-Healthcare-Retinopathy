from PIL import Image, ImageFilter
import os
import sys

import numpy as np
import cv2

if len(sys.argv) < 3 :
	print "Usage: resizeGscaleRgb.py <inputDir> <outputDir>"
	sys.exit()

outputDir = sys.argv[2]
inputDir = sys.argv[1]

filesList = os.listdir(inputDir)
#filesList.remove('gscale')

if not os.path.exists(outputDir):
	os.makedirs(outputDir)
#if not os.path.exists(outputDir + "AH"):
#        os.makedirs(outputDir + "AH")

img_rows, img_cols = 200, 200
clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))

for file in filesList:
	base = os.path.basename(inputDir + "/" + file)
	fileName = os.path.splitext(base)[0]
	print(fileName)
	im = Image.open(inputDir + "/" + file)
	img = im.resize((img_rows,img_cols)).convert('L').filter(ImageFilter.SHARPEN).save(outputDir + "/"  + file, 'jpeg')
	img2 = cv2.imread(outputDir + "/" + file, 0)
	cl1 = clahe.apply(img2)
	cv2.imwrite(outputDir + "/" + file, cl1)
	im = Image.open(outputDir + "/" + file)
        img = im.convert('RGB').save(outputDir + "/" + file,'jpeg')

