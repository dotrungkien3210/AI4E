import tensorflow as tf
import cv2
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras import backend as K

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train_grayscale = np.zeros(x_train.shape[:-1])
for i in range(x_train.shape[0]):
    x_train_grayscale[i] = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY)
x_train = x_train_grayscale

x_test_grayscale = np.zeros(x_test.shape[:-1])
for i in range(x_test.shape[0]):
    x_test_grayscale[i] = cv2.cvtColor(x_test[i], cv2.COLOR_BGR2GRAY)
x_test = x_test_grayscale
batch_size = 128
num_classes = 10
epochs = 12
x_train = x_train.reshape(-1, 1024) # N
x_test = x_test.reshape(-1, 1024)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize (0-1)
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(len(x_train))
print(len(y_train))
# convert class vectors to binary class matrices
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[0])

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1024,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(), # adam, .... gradient descent
              metrics=['accuracy'])
H = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])