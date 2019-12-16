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
from sklearn.model_selection import train_test_split
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def getVGGFeatures(fileList, layerName):
	#Initial Model Setup
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	
	#Confirm number of files passed is what was expected
	rArray = []
	print ("Number of Files Passed:")
	print(len(fileList))

	for iPath in fileList:
		#Time Printing for Debug, you can comment this out if you wish
		# now = datetime.now()
		# current_time = now.strftime("%H:%M:%S")
		# print("Current Time =", current_time)
		try:
			#Read Image
			img = image.load_img(iPath)
			#Update user as to which image is being processed
			print("Getting Features " +iPath)
			#Get image ready for VGG16
			img = img.resize((224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			#Generate Features
			internalFeatures = model.predict(x)
			rArray.append((iPath, internalFeatures))			
		except:
			print ("Failed "+ iPath)
	return rArray


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
	# turn images into greyscale
	preProcessedImages = np.asarray(list(map(lambda x: np.array(x.convert("L")), preProcessedImages)))

	# enumerate varible categories from string to int
	enum_labels = dict((y, x) for x, y in enumerate(set(labels)))
	labels = np.asarray([enum_labels[x] for x in labels])

	# set params
	batch_size = 16
	n_classes = len(enum_labels)
	epochs = 40
	img_rows, img_cols = 60, 60

	# shuffle data and split into train, test, validation
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
	
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)

	# building the input vector from the 60x60 pixels
	X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
	X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
	X_valid = X_valid.reshape(X_valid.shape[0], img_rows * img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')

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
	model.add(Dense(n_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

	# training the model and saving metrics in history
	history = model.fit(X_train, y_train,
			batch_size=batch_size, epochs=epochs,
			verbose=2, validation_data=(X_valid, y_valid))

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	
	# plot loss
	x = range(1, len(history.history["val_loss"]) + 1)
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.plot(x, history.history["val_loss"], label="validation loss")
	ax.plot(x, history.history["loss"] , label="training loss")
	ax.set_title('Model Loss')
	ax.set_ylabel('Loss')
	ax.set_xlabel('Epoch')

	plt.legend()

	ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig.savefig('plot1.png')


def trainFaceClassifier_VGG(extractedFeatures, labels):
	extractedFeatures = np.asarray([x[1].flatten() for x in extractedFeatures])

	# enumerate varible categories from string to int
	enum_labels = dict((y, x) for x, y in enumerate(set(labels)))
	labels = np.asarray([enum_labels[x] for x in labels])

	# set params
	batch_size = 32
	n_classes = len(enum_labels)
	epochs = 20

	# shuffle data and split into train, test, validation
	X_train, X_test, y_train, y_test = train_test_split(extractedFeatures, labels, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
	
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)


	# one-hot encoding using keras' numpy-related utilities
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	y_valid = keras.utils.to_categorical(y_valid, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)

	# building a linear stack of layers with the sequential model
	model = Sequential()
	model.add(Dense(512, input_shape=(X_train.shape[1],)))
	model.add(Activation('relu'))                            
	model.add(Dense(n_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

	# training the model and saving metrics in history
	history = model.fit(X_train, y_train,
			batch_size=batch_size, epochs=epochs,
			verbose=2, validation_data=(X_valid, y_valid))

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


def part4_dropout():
	directory = "cropped"
	preProcessedImages = [Image.open(directory + "/" + filename) for filename in os.listdir(directory)]
	labels = [re.search(r'^[^\d]*', filename).group() for filename in os.listdir(directory)]
	# turn images into greyscale
	preProcessedImages = np.asarray(list(map(lambda x: np.array(x.convert("L")), preProcessedImages)))

	# enumerate varible categories from string to int
	enum_labels = dict((y, x) for x, y in enumerate(set(labels)))
	labels = np.asarray([enum_labels[x] for x in labels])

	# set params
	batch_size = 32
	n_classes = len(enum_labels)
	epochs = 40
	img_rows, img_cols = 60, 60

	# shuffle data and split into train, test, validation
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
	
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)

	# building the input vector from the 60x60 pixels
	X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
	X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
	X_valid = X_valid.reshape(X_valid.shape[0], img_rows * img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')

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

	accuracy_scores = []
	loss_scores = []
	dropout_rates = np.arange(0, 0.9, 0.05)

	for i in dropout_rates:
		# building a linear stack of layers with the sequential model
		model = Sequential()
		model.add(Dense(512, input_shape=(img_rows*img_cols,)))
		model.add(Dropout(rate=i))
		model.add(Activation('relu'))                            
		model.add(Dense(n_classes))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

		# training the model and saving metrics in history
		model.fit(X_train, y_train,
				batch_size=batch_size, epochs=epochs,
				verbose=0, validation_data=(X_valid, y_valid))

		score = model.evaluate(X_test, y_test, verbose=0)
		accuracy_scores.append(score[1])
		loss_scores.append(score[0])
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		print("Dropout Rate:", i)

	
	# plot
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.plot(dropout_rates, loss_scores, label="Test loss")
	ax.plot(dropout_rates, accuracy_scores, label="Test accuracy")
	ax.set_title('Model Dropout')
	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Dropout Rate')

	plt.legend()

	ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig.savefig('plot4.png')


def part4_dropout_VGG():
	directory = "cropped"
	images = [directory + "/" + filename for filename in os.listdir(directory)]
	extractedFeatures = getVGGFeatures(images, "block4_pool")
	labels = [re.search(r'^[^\d]*', filename).group() for filename in os.listdir(directory)]
	extractedFeatures = np.asarray([x[1].flatten() for x in extractedFeatures])

	# enumerate varible categories from string to int
	enum_labels = dict((y, x) for x, y in enumerate(set(labels)))
	labels = np.asarray([enum_labels[x] for x in labels])

	# set params
	batch_size = 32
	n_classes = len(enum_labels)
	epochs = 10

	# shuffle data and split into train, test, validation
	X_train, X_test, y_train, y_test = train_test_split(extractedFeatures, labels, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
	
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)


	# one-hot encoding using keras' numpy-related utilities
	print("Shape before one-hot encoding: ", y_train.shape)
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	y_valid = keras.utils.to_categorical(y_valid, n_classes)
	print("Shape after one-hot encoding: ", y_train.shape)


	accuracy_scores = []
	loss_scores = []
	dropout_rates = np.arange(0, 0.9, 0.05)


	for i in dropout_rates:
		# building a linear stack of layers with the sequential model
		model = Sequential()
		model.add(Dense(512, input_shape=(X_train.shape[1],)))
		model.add(Dropout(rate=i))
		model.add(Activation('relu'))                            
		model.add(Dense(n_classes))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

		# training the model and saving metrics in history
		model.fit(X_train, y_train,
				batch_size=batch_size, epochs=epochs,
				verbose=0, validation_data=(X_valid, y_valid))

		score = model.evaluate(X_test, y_test, verbose=0)
		accuracy_scores.append(score[1])
		loss_scores.append(score[0])
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		print("Dropout Rate:", i)

	
	# plot
	fig, ax = plt.subplots(figsize=(12, 8))
	# ax.plot(dropout_rates, loss_scores, label="Test loss")
	ax.plot(dropout_rates, accuracy_scores, label="Test accuracy")
	ax.set_title('Model Dropout VGG')
	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Dropout Rate')

	plt.legend()

	ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig.savefig('plot3.png')

	fig1, ax1 = plt.subplots(figsize=(12, 8))
	ax1.plot(dropout_rates, loss_scores, label="Test loss")
	# ax1.plot(dropout_rates, accuracy_scores, label="Test accuracy")
	ax1.set_title('Model Dropout VGG')
	ax1.set_ylabel('Accuracy')
	ax1.set_xlabel('Dropout Rate')

	plt.legend()

	ax1.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig1.savefig('plot4.png')


	fig2, ax2 = plt.subplots(figsize=(12, 8))
	ax2.plot(dropout_rates, loss_scores, label="Test loss")
	ax2.plot(dropout_rates, accuracy_scores, label="Test accuracy")
	ax2.set_title('Model Dropout VGG')
	ax2.set_ylabel('Accuracy')
	ax2.set_xlabel('Dropout Rate')

	plt.legend()

	ax2.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig2.savefig('plot5.png')


def part4_layers():
	directory = "cropped"
	preProcessedImages = [Image.open(directory + "/" + filename) for filename in os.listdir(directory)]
	labels = [re.search(r'^[^\d]*', filename).group() for filename in os.listdir(directory)]
	# turn images into greyscale
	preProcessedImages = np.asarray(list(map(lambda x: np.array(x.convert("L")), preProcessedImages)))

	# enumerate varible categories from string to int
	enum_labels = dict((y, x) for x, y in enumerate(set(labels)))
	labels = np.asarray([enum_labels[x] for x in labels])

	# set params
	batch_size = 32
	n_classes = len(enum_labels)
	epochs = 40
	img_rows, img_cols = 60, 60

	# shuffle data and split into train, test, validation
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
	
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)

	# building the input vector from the 60x60 pixels
	X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
	X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
	X_valid = X_valid.reshape(X_valid.shape[0], img_rows * img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')

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

	accuracy_scores = []
	loss_scores = []
	num_layers = range(1, 11)

	for i in num_layers:
		# building a linear stack of layers with the sequential model
		model = Sequential()
		model.add(Dense(512, input_shape=(img_rows*img_cols,)))
		model.add(Activation('relu'))        
		for _ in range(i+1):                    
			model.add(Dense(n_classes))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

		# training the model and saving metrics in history
		model.fit(X_train, y_train,
				batch_size=batch_size, epochs=epochs,
				verbose=0, validation_data=(X_valid, y_valid))

		score = model.evaluate(X_test, y_test, verbose=0)
		accuracy_scores.append(score[1])
		loss_scores.append(score[0])
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

	
	# plot
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.plot(num_layers, loss_scores, label="Test loss")
	ax.plot(num_layers, accuracy_scores, label="Test accuracy")
	ax.set_title('Model Hidden Layers')
	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Number Hidden Layers')

	plt.legend()

	ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig.savefig('plot4.png')



if __name__ == '__main__':
	print("A5")
	# -----Part 1-----
	# preProcessImages()

	# -----Part 2-----
	# grabs all files in given directory and gets images and image label (letters up till first digit)
	# directory = "cropped"
	# preProcessedImages = [Image.open(directory + "/" + filename) for filename in os.listdir(directory)]
	# labels = [re.search(r'^[^\d]*', filename).group() for filename in os.listdir(directory)]
	# trainFaceClassifier(preProcessedImages, labels)

	# -----Part 3-----
	# directory = "cropped"
	# images = [directory + "/" + filename for filename in os.listdir(directory)]
	# features = getVGGFeatures(images, "block4_pool")
	# labels = [re.search(r'^[^\d]*', filename).group() for filename in os.listdir(directory)]
	# trainFaceClassifier_VGG(features, labels)

	# -----Part 4-----
	# part4_dropout()
	# part4_dropout_VGG()
	# part4_layers()

	# -----Part 5-----
