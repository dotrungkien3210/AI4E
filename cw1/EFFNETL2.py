import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow import keras
from keras.datasets import cifar100
from  skimage import transform
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


model = EfficientNetB7(include_top=False, weights='imagenet')
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_img = np.array((50000, 224, 224, 3))
for a in range(len(x_train)):
    train_img[a] = cv2.resize(x_train[a],None,fx=7,fy=7, interpolation=cv2.INTER_CUBIC)
print(train_img[0].shape)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255
x_val = x_train[40000:50000,:]
x_train = x_train[0:40000,:]
y_val = y_train[40001:50000,:]
y_train = y_train[0:40000,:]
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, 100)
y_val = keras.utils.to_categorical(y_val, 100)
y_test = keras.utils.to_categorical(y_test, 100)

'''n_classes = 100
epochs = 15
batch_size = 8
eff7 = EfficientNetB7(include_top=False, weights='imagenet',input_shape=(224,224,3), classes=n_classes)

model = Sequential()
model.add(eff7)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.summary()'''