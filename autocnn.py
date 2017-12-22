from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.callbacks import TensorBoard
from dataset import stl10_input as stl10
import h5py
import pickle
from utils import utils

AUTOENC_WEIGHTS= './output/models/stl10_autoenc_weights.hdf5'
AUTOENC_ARCH = './output/models/stl10_autoenc_arch.json'
MODEL_WEIGHTS = './output/models/stl10_model_weights.hdf5'
MODEL_ARCH = './output/models/stl10_model_arch.json'
MODELS_DIR = './output/models/'

MAX_UNLAB = 25000
MAX_TEST = 8000
MAX_TRAIN = 5000
print("[INFO] loading STL-10 data...")
stl10.download_and_extract()

input_img = Input(shape=(96, 96, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (24,24, 16) i.e. 9216-dimensional

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

# set up the FUlly connected model for final training on labelled data
x = autoencoder.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)
model_final = Model(inputs = autoencoder.input, outputs = predictions)

# set up test values used to evaluate final model
x_test = stl10.read_all_images(stl10.TEST_DATA_PATH, MAX_TEST)
x_test = x_test[:,:,:,:].astype("float") / 255.0
testY = stl10.read_labels(stl10.TEST_LABEL_PATH)[0:MAX_TEST]
lb = LabelBinarizer()
y_test = lb.fit_transform(testY)

# train or preload final model
if os.path.exists(MODEL_WEIGHTS): 
	model_final.load_weights(MODEL_WEIGHTS, by_name = True)
	with open(MODELS_DIR + 'history.pickle', 'rb') as f:
		model_history = pickle.load(f)
else:
	# train or preload the autoencoded weights
	if not os.path.exists(AUTOENC_WEIGHTS):  
		x_unlabeled = stl10.read_all_images(stl10.UNLABELED_DATA_PATH, MAX_UNLAB)
		x_unlabeled = x_unlabeled[:,:,:,:].astype("float") / 255.0
		hist1 =autoencoder.fit(x_unlabeled, x_unlabeled,
                	epochs=25,
               	batch_size=128,
                	shuffle=True
               	)
		
		# free some memory
		del x_unlabeled
		# keep decoded autoencoded images for examination
		decoded_imgs = autoencoder.predict(x_test)

		model_final.set_weights(autoencoder.get_weights())
		# serialize weights to HDF5
		autoencoder.save_weights(AUTOENC_WEIGHTS, overwrite=True)
		print("Saved model to disk")
		# serialize history
		with open(MODELS_DIR + 'hist1.pickle', 'wb') as f:
			pickle.dump(hist1.history, f)
	else:
		model_final.load_weights(AUTOENC_WEIGHTS, by_name = True)
	#
	# set up the final supervised model with the preinitialised weights 
	for layer in model_final.layers[:-7]:
		layer.trainable = False
	model_final.compile(optimizer='adadelta', loss='categorical_crossentropy')
	x_train = stl10.read_all_images(stl10.TRAIN_DATA_PATH, MAX_TRAIN)
	trainY = stl10.read_labels(stl10.TRAIN_LABEL_PATH)[0:MAX_TRAIN]
	x_train = x_train[:,:,:,:].astype("float") / 255.0
	y_train = lb.fit_transform(trainY)
	#train the final model
	history = model_final.fit(x_train, y_train,
                	epochs=50,
                	batch_size=128,
                	shuffle=True,
                	validation_data=(x_test, y_test))
	model_history = history.history
	# free some memory
	del x_train
	# serialize model history
	with open(MODELS_DIR  + 'history.pickle', 'wb') as f:
		pickle.dump(history.history, f)
	# serialize model to JSON
	model_json = model_final.to_json()
	with open(MODEL_ARCH, "w") as json_file:
    		json_file.write(model_json)
	# serialize weights to HDF5
	model_final.save_weights(MODEL_WEIGHTS)
	print("Saved model to disk")

with open(MODELS_DIR + 'hist1.pickle', 'rb') as f:
	autoenc_history = pickle.load(f)

labelNames = ["airplane", "bird", "car", "cat", "deer",
 "dog", "horse", "monkey", "ship", "truck"]

final_predictions = utils.decode_predictions(model_final.predict(x_test),labelNames)
for i in range(0,50):
	x = list(y_test[i]).index(True)
	print("predicted " + final_predictions[i][0] + " " + final_predictions[i][1] + " " + final_predictions[i][2] + " actual " + labelNames[list(y_test[i]).index(True)])

# plot loss curves
utils.plot_loss(model_history)
utils.plot_loss(autoenc_history)

# display decoded autoencode images if first autoencode training
if 'decoded_imgs' in globals():
	n = 10
	plt.figure(figsize=(20, 4))
	for i in range(1, n):
        		# display original
        		ax = plt.subplot(2, n, i)
        		plt.imshow(x_test[i].reshape(96,96,3))
        		#plt.gray()
        		ax.get_xaxis().set_visible(False)
        		ax.get_yaxis().set_visible(False)
        		# display reconstruction
        		ax = plt.subplot(2, n, i + n)
        		plt.imshow(decoded_imgs[i].reshape(96,96,3))
        		#plt.gray()
        		ax.get_xaxis().set_visible(False)
        		ax.get_yaxis().set_visible(False)

	plt.show()
