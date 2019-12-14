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
	directory = "uncropped"
	if not os.path.exists(directory):
    	os.makedirs(directory)
	for filename in os.listdir(directory):
		try:
			image = Image.open(directory + "/"+filename)
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
	
	batch_size = 128
	n_classes = 10
	epochs = 12

	# input image dimensions
	img_rows, img_cols = 60, 60

	# the data, split between train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)

	# building the input vector from the 28x28 pixels
	X_train = X_train.reshape(60000, img_rows*img_cols)
	X_test = X_test.reshape(10000, img_rows*img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# normalizing the data to help with the training
	X_train /= 255
	X_test /= 255

	# print the final input shape ready for training
	print("Train matrix shape", X_train.shape)
	print("Test matrix shape", X_test.shape)



	# one-hot encoding using keras' numpy-related utilities
	print("Shape before one-hot encoding: ", y_train.shape)
	Y_train = keras.utils.to_categorical(y_train, n_classes)
	Y_test = keras.utils.to_categorical(y_test, n_classes)
	print("Shape after one-hot encoding: ", Y_train.shape)



	# building a linear stack of layers with the sequential model
	model = Sequential()
	model.add(Dense(512, input_shape=(img_rows*img_cols,)))
	model.add(Activation('relu'))                            

	model.add(Dense(10))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


	# training the model and saving metrics in history
	history = model.fit(X_train, Y_train,
			batch_size=128, epochs=20,
			verbose=2,
			validation_data=(X_test, Y_test))


	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	# split into training, validation and test data


def trainFaceClassifier_VGG(extractedFeatures, labels):
	utils.raiseNotDefined()


if __name__ == '__main__':
	print("Your Program Here")
	preProcessImages()