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



import numpy as np

from PIL import Image

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
	utils.raiseNotDefined()

def standardizeImage(image, x, y):
	utils.raiseNotDefined()

def preProcessImages(images):
	utils.raiseNotDefined()

def visualizeWeight():
	#You can change these parameters if you need to
	utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
	utils.raiseNotDefined()

def trainFaceClassifier_VGG(extractedFeatures, labels):
	utils.raiseNotDefined()


if __name__ == '__main__':
	print("Your Program Here")