from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

image_path = r"/home/nmelgiri/Downloads/autodraw 10_03_2019 (2).png"

model = load_model('results/keras_mnist.h5')
image_input = image.img_to_array(image.load_img(path=image_path, grayscale=True, target_size=(28, 28, 1)))
image_input = (255 - image_input) / 255
image_input = image_input.reshape((1, 784))

print(model.predict(image_input))
