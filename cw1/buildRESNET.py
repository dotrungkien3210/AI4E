import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
from keras.datasets import cifar100
import numpy as np
from keras import layers
from keras.layers import Input, Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D
from keras.models import Model,load_model
from keras.initializers import glorot_uniform
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
rows = 64
cols = 64
channels = 3
classes = 2 # phân loại chó mèo
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255
x_val = x_train[40000:50000,:]
x_train = x_train[0:40000,:]
y_val = y_train[40001:50000,:]
y_train = y_train[0:40000,:]
y_train = keras.utils.to_categorical(y_train, 100)
y_val = keras.utils.to_categorical(y_val, 100)
y_test = keras.utils.to_categorical(y_test, 100)
x = x_train[0]
print(x.shape)
resize = tf.keras.layers.UpSampling2D(size=(7,7))(x)