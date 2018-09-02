import keras
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.models import Sequential, load_model
from random import randint
from keras.utils import plot_model

test_name = 'cnn2'
count = 6000
epoch = 5
rand = randint(0, 60000-count)
count = rand + count

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

try:
    model = load_model('%s.hdf5' % test_name)
except Exception as e:
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=(28, 28, 1),
                     name='C1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='S1'))
    model.add(Conv2D(6, (3, 3), activation='relu', name='C2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S2'))
    model.add(Conv2D(6, (3, 3), activation='relu', name='C3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S3'))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


model.fit(x_train[rand:count], y_train[rand:count], epochs=epoch)

model.save('%s.hdf5' % test_name)

print(model.evaluate(x_test, y_test))
print(model.metrics_names)
