from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.transform import rescale, resize
import math
from skimage.morphology import skeletonize
from skimage.measure import moments
from skimage import morphology
from scipy.ndimage import rotate
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	for i in [9]:
		trainX = np.delete(trainX, np.where(trainY ==i), axis=0)
		trainY = np.delete(trainY, np.where(trainY==i))
		testX = np.delete(testX, np.where(testY ==i), axis=0)
		testY = np.delete(testY, np.where(testY==i))
	trainX = trainX.astype('float32')
	testX = testX.astype('float32')
	#Create two new datasets for the rotated digits
	#reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	print(trainY.shape)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

def define_model():
	model = Sequential()   
	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())    
	model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(512,activation="relu"))
	model.add(Dense(9,activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model
	
def conf_matrix():
	model = define_model()
	model.load_weights("weights/best_norot.hdf5")
	print(model.summary())
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	trainX, trainY, testX, testY = load_dataset()
	trainX, testX = prep_pixels(trainX, testX)
	fig = plt.figure(figsize=(10, 10)) # Set Figure
	y_pred = model.predict(testX) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels
	Y_test = np.argmax(testY, 1) # Decode labels
	mat = confusion_matrix(Y_test, Y_pred, normalize='true') # Confusion matrix
	# Plot Confusion matrix
	sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values');
	plt.savefig('confusion_matrix_norot.pdf')
	plt.show();


# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	##path for the weights
	filepath = "weights/best_norot.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	# Fit the model
	model.fit(trainX, trainY, validation_data=(testX, testY), epochs=30, batch_size=64, callbacks=callbacks_list, verbose=1)

def eval_model():
	# define model
	model = define_model()
	# load model
	model = load_model('otherothermodel_epochs20_order3_axis_batch64.h5')
	acc_list = []
	for angle in range(0,90,30):
		trainX, trainY, testX, testY = load_dataset()
		# prepare pixel data
		trainX, testX = prep_pixels(trainX, testX)
		# evaluate model on test dataset
		_, acc = model.evaluate(testX, testY, verbose=0)
		acc_list.append(acc)
		print('> %.3f' % (acc * 100.0))
	fig, ax = plt.subplots(1,1, figsize=(16, 10))
	ax.plot(acc_list)
	plt.show()
	

run_test_harness()
conf_matrix()
#eval_model()
