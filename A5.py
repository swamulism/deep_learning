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
import re

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
	directory = "cropped"
	if not os.path.exists(directory):
		os.makedirs(directory)
	for filename in os.listdir("uncropped"):
		try:
			image = Image.open("uncropped/" + filename)
			thing = filename.split("-")
			cords, extention = thing[1].split(".")
			cords = map(int, cords.split(","))

			image = cropImage(image, *cords)
			image = standardizeImage(image, 60, 60)

			filename = thing[0] + "." + extention
			image.save(directory + "/" + filename)

			print(f"{filename} success")
		except:
			print(f"{filename} fail")

def visualizeWeight():
	#You can change these parameters if you need to
	utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
	preProcessedImages = map(lambda x: x.convert("L"), preProcessedImages)

	preProcessedImages, labels = zip(zip(preProcessedImages, labels).shuffle())
	
	batch_size = 32
	n_classes = len(set(labels))
	epochs = 20

	# input image dimensions
	img_rows, img_cols = 60, 60

	# the data, split between train and test sets
	X_train, X_test = split_data(preProcessedImages, 0.2)
	X_train, X_valid = split_data(X_train, 0.2)

	y_train, y_test = split_data(labels, 0.2)
	y_train, y_valid = split_data(y_train, 0.2)

	# building the input vector from the 60x60 pixels
	X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
	X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
	X_valid = X_valid.reshape(X_test.shape[0], img_rows * img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_vaild = X_vaild.astype('float32')

	# normalizing the data to help with the training
	X_train /= 255
	X_test /= 255
	X_valid /= 255

	# one-hot encoding using keras' numpy-related utilities
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	y_valid = keras.utils.to_categorical(y_valid, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)

	# building a linear stack of layers with the sequential model
	model = Sequential()
	model.add(Dense(512, input_shape=(img_rows*img_cols,)))
	model.add(Activation('relu'))                            

	model.add(Dense(10))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


	# training the model and saving metrics in history
	history = model.fit(X_train, y_train,
			batch_size=batch_size, epochs=epochs,
			verbose=2, validation_data=(X_valid, y_valid))


	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	# split into training, validation and test data


def trainFaceClassifier_VGG(extractedFeatures, labels):
	utils.raiseNotDefined()

def split_data(data, split):
	return data[:int(len(data) * (1 - split))], data[int(len(data) * (1 - split)):]

if __name__ == '__main__':
	print("Your Program Here")
	# Part 1
	# preProcessImages()

	# Part 2
	# preProcessedImages = [Image.open(filename) for filename in os.listdir("cropped")]
	# labels = [re.search(r'^[^\d]*', filename).group() for filename in os.listdir("cropped")]
	# trainFaceClassifier(preProcessedImages, labels)