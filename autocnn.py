from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from dataset import stl10_input as stl10

print("[INFO] loading STL-10 data...")
stl10.download_and_extract()
#with open(stl10.UNLABELED_DATA_PATH) as sf:
#        unlabeledX = read_single_image(sf)
unlabeledX = stl10.read_all_images(stl10.UNLABELED_DATA_PATH)

trainX = stl10.read_all_images(stl10.TRAIN_DATA_PATH)

trainY = stl10.read_labels(stl10.TRAIN_LABEL_PATH)

testX = stl10.read_all_images(stl10.TEST_DATA_PATH)

testY = stl10.read_labels(stl10.TEST_LABEL_PATH)

x_unlabeled = np.empty(stl10.SSHAPE, dtype=float)
x_unlabeled = unlabeledX[0:stl10.SSHAPE[0],:,:,:].astype("float") / 255.0
del unlabeledX

x_train = np.empty(stl10.SSHAPE, dtype=float)
x_train = trainX[0:stl10.SSHAPE[0],:,:,:].astype("float") / 255.0
del trainX
x_test = np.empty(stl10.SSHAPE, dtype=float)
x_test = testX[0:stl10.SSHAPE[0],:,:,:].astype("float") / 255.0
del testX
lb = LabelBinarizer()
y_train = lb.fit_transform(trainY)[0:stl10.SSHAPE[0]]
del trainY
y_test = lb.transform(testY)[0:stl10.SSHAPE[0]]
del testY

labelNames = ["airplane", "bird", "car", "cat", "deer",
 "dog", "horse", "monkey", "ship", "truck"]

input_img = Input(shape=(96, 96, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (12,12, 8) i.e. 1152-dimensional

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#autoencoder.fit(x_unlabeled, x_unlabeled,
#                epochs=50,
 #               batch_size=128,
  #              shuffle=True,
   #             validation_data=(x_test, x_test))
#,
#               callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
#open(stl10.ARCH_OUT, "w").write(autoencoder.to_json())
#autoencoder.save_weights(stl10.WEIGHTS_OUT, overwrite=True)

x = autoencoder.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)
model_final = Model(inputs = autoencoder.input, outputs = predictions)

if not os.path.exists(stl10.WEIGHTS_OUT):
	model_final.layers[-6].set_weights(autoencoder.output.get_weights())
	autoencoder.save_weights(stl10.WEIGHTS_OUT, overwrite=True)
else:
	model_final.load_weights(stl10.WEIGHTS_OUT, by_name = True)

for layer in autoencoder.layers[:-1]:
	layer.trainable = False
model_final.compile(optimizer='adadelta', loss='categorical_crossentropy')

history = model_final.fit(x_train, y_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, y_test))

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

decoded_imgs = autoencoder.predict(x_test)

final_predictions = decode_predictions(final_model.predict(x_test),labelNames)
print(final_predictions[:,:10])
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
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
   	 ax = plt.subplot(1, n, i)
   	 plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
   	 plt.gray()
   	 ax.get_xaxis().set_visible(False)
   	 ax.get_yaxis().set_visible(False)
plt.show()
