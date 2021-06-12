import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

x_train = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5],dtype= float)
y_train = np.array([30,21,14,9,6,5,6,9,14,21,30],dtype= float)
x_test =  np.arange(-50,51)
model = keras.Sequential([
keras.layers.Dense(units=9,input_shape=[1]),
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=500)
y_test = model.predict(x_test)
print(y_test)
#plt.figure(1)
#plt.plot(x_train,y_train)
#plt.figure(2)
#plt.plot(x_test,y_test)
plt.show()