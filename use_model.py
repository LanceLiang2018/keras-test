import keras
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.models import Sequential, load_model
from keras.utils import plot_model

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = load_model('model2.hdf5')

print(model.evaluate(x_test, y_test))
print(model.metrics_names)

