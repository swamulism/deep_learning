from __future__ import print_function
import keras
import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import os


import numpy as np

from PIL import Image


# def getVGGFeatures(fileList, layerName):
# 	#Initial Model Setup
# 	base_model = VGG16(weights='imagenet')
# 	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	
# 	#Confirm number of files passed is what was expected
# 	rArray = []
# 	print ("Number of Files Passed:")
# 	print(len(fileList))

# 	for iPath in fileList:
# 		#Time Printing for Debug, you can comment this out if you wish
# 		now = datetime.now()
# 		current_time = now.strftime("%H:%M:%S")
# 		print("Current Time =", current_time)
# 		try:
# 			#Read Image
# 			img = image.load_img(iPath)
# 			#Update user as to which image is being processed
# 			print("Getting Features " +iPath)
# 			#Get image ready for VGG16
# 			img = img.resize((224, 224))
# 			x = image.img_to_array(img)
# 			x = np.expand_dims(x, axis=0)
# 			x = preprocess_input(x)
# 			#Generate Features
# 			internalFeatures = model.predict(x)
# 			rArray.append((iPath, internalFeatures))			
# 		except:
# 			print ("Failed "+ iPath)
# 	return rArray

def getVGGFeatures(img, layerName):
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	img = img.resize((224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	internalFeatures = model.predict(x)

	return internalFeatures

def cropImage(image, x1, y1, x2, y2):
	return image.crop((x1, y1, x2, y2))

def standardizeImage(image, x, y):
	return image.resize((x, y))

def preProcessImages(images=None):
	for filename in os.listdir("uncropped"):
		try:
			image = Image.open("uncropped/"+filename)
			thing = filename.split("-")
			cords, extention = thing[1].split(".")
			cords = map(int, cords.split(","))

			image = cropImage(image, *cords)
			image = standardizeImage(image, 60, 60)

			filename = thing[0] + "." + extention
			image.save("cropped/"+filename)

			print(f"{filename} success")
		except:
			print(f"{filename} fail")

def visualizeWeight():
	#You can change these parameters if you need to
	utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
	utils.raiseNotDefined()
	# img = ​Image​.open(​'image.png'​).convert(​'L'​)

def trainFaceClassifier_VGG(extractedFeatures, labels):
	utils.raiseNotDefined()


if __name__ == '__main__':
	print("Your Program Here")
	preProcessImages()