import os

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # Load the MNIST dataset
X_train = (X_train.reshape(60000, 784) / 255).astype('float32')
X_test = (X_test.reshape(10000, 784) / 255).astype('float32')
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(16, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
model.fit(X_train, Y_train, batch_size=128, epochs=30, verbose=2, validation_data=(X_test, Y_test))

# saving the model
save_dir = "results"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
