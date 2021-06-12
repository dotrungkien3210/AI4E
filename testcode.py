import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10


'''(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255
x_val = x_train[40000:50000,:]
x_train = x_train[0:40000,:]
y_val = y_train[40001:50000,:]
y_train = y_train[0:40000,:]
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train[0].shape)
y = tf.keras.layers.UpSampling3D(size=5)(x_train[0])

print(y)'''
input_shape = (2, 1, 2, 1, 3)
x = tf.constant(1, shape=input_shape)
y = tf.keras.layers.UpSampling3D(size=2)(x)
print(y.shape)
