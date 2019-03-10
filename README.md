# Machine Learning Tutorial / Keras-Tensorflow-MNIST
## Machine Learning and Neural Networks
Similar to how driving is a method of transportation and getting a degree in humanities is a method of lifelong underemployment, a Neural Network is a method of Machine Learning. Watch the four videos by my favourite YouTuber- [3blue1brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw). I will supplement each video with my thoughts to help you prioritise your attention.

  

### CHAPTER 1: But what *is* a Neural Network?

 
This chapter of the Deep Learning series explores the topic of neural networks jumping right in with the math. Each individual comment said isn't "hard" to understand, as it only uses basic algebra. But the whole concept of neural networks may be a bit tricky to understand with the math dabbled in so often. I don't want you to get lost understanding every small technical detail in this. I do not want you to spend hours understanding this video. Just play it once or twice and develop a basic "feeling" for these networks. Similar to how you develop a "feeling" for riding a bike before going fast and doing cool stuff with it. I want you to come out with the following learning outcomes:
<ul>
	<li>How does every neuron know what to do? Don't focus 100% on the math, focus on the messages being sent to each neuron.</li>
   <li>How do all these different weights together get us closer to identifying shapes? [focus on concept, not so much on the math]</li>
</ul>
aka. Don't focus on the math so much. We'll get to that part later after we get our first program done. Watch and enjoy!

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/aircAruvnKk/0.jpg)](https://www.youtube.com/watch?v=aircAruvnKk)


### CHAPTER 2: Gradient Decent: how neural networks learn?

  

Chapter two of the Deep Learning series focuses on how an algorithm retraces it's steps and figures out it's mistakes. The math starts to get really intense here. Just go into this video with the concept of derivatives in your head. By changing a weight that was wrong by a small amount, how big of an effect would that have? Imagine a parabola- if every derivative was an arrow / vector, then the derivative always points in the general direction toward the minimum. The steepness of it tells you how much how big of an effect each change would have. I want you to come out with the following learning outcomes:
<ul>
	<li>How does every neuron know what to do? Don't focus 100% on the math, focus on the messages being sent to each neuron.</li>
	<li>How do all these different weights together get us closer to identifying shapes? [focus on concept, not so much on the math]</li>
</ul>
aka. Don't focus on the math so much. We'll get to that part later. Watch and enjoy!

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/IHZwWFHWa-w/0.jpg)](https://www.youtube.com/watch?v=IHZwWFHWa-w)





##
## TIME TO CODE!!!
### Necessary Libraries

```python
from os.path import join  
  
import matplotlib.pyplot as plt  # MatPlotLib is used to display the dataset we have  
from tensorflow.python.keras.datasets import mnist  # The MNIST Dataset has the pre-labeled handwriting dataset  
from tensorflow.python.keras.layers import Dense, Activation, Dropout  
from tensorflow.python.keras.models import Sequential  
from tensorflow.python.keras.optimizers import SGD  
from tensorflow.python.keras.utils import to_categorical
```

### Load MNIST Dataset
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
### Display first four elements of the MNIST Dataset
```python
plt.subplot(221)  
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))  
plt.subplot(222)  
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))  
plt.subplot(223)  
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))  
plt.subplot(224)  
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))  
plt.show()
```
### Reshape the Dataset
```python
# building the input vector from the 28x28 pixels. In the video tutorial, the input was condensed to a straight line  
X_train = X_train.reshape(60000, 784) # 6000 images, 784 pixels per image  
X_test = X_test.reshape(10000, 784) # 10000 images, 784 pixels per image  
X_train = X_train.astype('float32')  
X_test = X_test.astype('float32')  
  
# normalizing the data to help with the training; all values and inputs have to be between 0 and 1  
X_train /= 255  
X_test /= 255
n_classes = 10
Y_train = to_categorical(y_train, n_classes)  
Y_test = to_categorical(y_test, n_classes)
```
### Design the Neural Network
```python
# Input layer: input layer with 784 pixels connecting to a hidden layer of 16 neurons (Keras can determine the weights)  
model = Sequential() 						# The traditional Neural Network discussed  
model.add(Dense(16, input_shape=(784,))) 	# Input Layer connecting with a hidden Layer  
model.add(Activation('relu')) 				# Activation function to compress 0<result<1  
model.add(Dropout(0.2)) 					# To prevent over-fitting; will explain in  
												# a future tutorial  
  
model.add(Dense(16)) 						# 2nd Hidden Layer  
model.add(Activation('relu')) 				# Activation function  
model.add(Dropout(0.2)) 					# help prevent over-fitting  
  
model.add(Dense(10)) 						# Output layer; has 10 possible outputs  
model.add(Activation('softmax')) 			# Activation function  
  
model.compile(  
  loss='categorical_crossentropy', 			# (result - loss)^2  
  metrics=['accuracy'], 					# Display Accuracy while training  
  optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
											# The optimizer as discussed in the video  
)
```
### Run and Store the model results
```python
history = model.fit(X_train, Y_train, batch_size=128, epochs=30, verbose=2, validation_data=(X_test, Y_test))

# saving the model  
save_dir = "results"  
model_name = 'keras_mnist.h5'  
model_path = join(save_dir, model_name)  
model.save(model_path)  
print('Saved trained model at %s ' % model_path)
```
### Displaying the Model Results
```python
  
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
```