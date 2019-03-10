from os.path import join

import matplotlib.pyplot as plt  # MatPlotLib is used to display the dataset we have
from tensorflow.python.keras.datasets import mnist  # The MNIST Dataset has the pre-labeled handwriting dataset
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import to_categorical

'''------------------------------------------------------------------------------------------------------------------'''
# We need to first get the Hand Written DataSet
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # Load the MNIST dataset

'''------------------------------------------------------------------------------------------------------------------'''
'''********************************************[DISPLAY FIRST 4 ELEMENTS]********************************************'''
'''------------------------------------------------------------------------------------------------------------------'''
# To get a feel of what we're dealing with, let's display the first four elements
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()

'''------------------------------------------------------------------------------------------------------------------'''
'''********************************************[RESHAPE THE IMAGE + SHOW]********************************************'''
'''------------------------------------------------------------------------------------------------------------------'''
# let's print the shape before we reshape and normalize; understand what we're doing to the data
print()
print("TRAINING DATASET INFORMATION")
print("X_train shape: (number of training samples, image width, image height) : ", X_train.shape)
print("y_train shape: (number of training answers)                            : ", y_train.shape)

print()
print("TESTING DATASET INFORMATION")
print("X_test shape: (number of testing samples, image width, image height) : ", X_test.shape)
print("y_test shape: (number of testing answers)                            : ", y_test.shape)
print()

# building the input vector from the 28x28 pixels. In the video tutorial, the input was condensed to a straight line
X_train = X_train.reshape(60000, 784)  # 6000 images, 784 pixels per image
X_test = X_test.reshape(10000, 784)  # 10000 images, 784 pixels per image
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training; all values and inputs have to be between 0 and 1
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape                                                   : ", X_train.shape)
print("Test matrix shape                                                    : ", X_test.shape)
print()

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
print()

'''------------------------------------------------------------------------------------------------------------------'''
'''**********************************************[BUILD NEURAL NETWORK]**********************************************'''
'''------------------------------------------------------------------------------------------------------------------'''
# This model should exactly mirror the ones shown in the videos by 3blue1brown

# Input layer: input layer with 784 pixels connecting to a hidden layer of 16 neurons (Keras can determine the weights)
model = Sequential()  # The traditional Neural Network discussed
model.add(Dense(16, input_shape=(784,)))  # Input Layer connecting with a hidden Layer
model.add(Activation('relu'))  # Activation function to compress 0<result<1
model.add(Dropout(0.2))  # To prevent over-fitting; will explain in
# a future tutorial

model.add(Dense(16))  # 2nd Hidden Layer
model.add(Activation('relu'))  # Activation function
model.add(Dropout(0.2))  # help prevent over-fitting

model.add(Dense(10))  # Output layer; has 10 possible outputs
model.add(Activation('softmax'))  # Activation function

model.compile(
	loss='categorical_crossentropy',  # (result - loss)^2
	metrics=['accuracy'],  # Display Accuracy while training
	optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)  # The optimizer as discussed in the video
)

print("SUMMARY OF MODEL: ")
model.summary()  # Display the current model
print()

'''------------------------------------------------------------------------------------------------------------------'''
'''*********************************************[TRAIN THE KERAS MODEL!]*********************************************'''
'''------------------------------------------------------------------------------------------------------------------'''

# training the model and saving metrics in history
history = model.fit(X_train, Y_train, batch_size=128, epochs=30, verbose=2, validation_data=(X_test, Y_test))

'''------------------------------------------------------------------------------------------------------------------'''
'''********************************************[DISPLAY TRAINING RESULTS]********************************************'''
'''------------------------------------------------------------------------------------------------------------------'''

# saving the model
save_dir = "results"
model_name = 'keras_mnist.h5'
model_path = join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# plotting the metrics
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()
