import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_square.csv')
N = data.shape[0]
x = data.iloc[:, 0]
x = x.values.reshape(-1, 1)
y = data.iloc[:, 1]
y = y.values.reshape(-1, 1)
def predict(X,theta):
    return X @ theta
xHeight = len(x)
xWidth = len(x[0])
theta = np.zeros((3,1))
xNEW = np.zeros((xHeight,3))
#print(x)
xNEW[:,0]= 1
xNEW[:,1]= x[:,0]
xNEW[:,2]= x[:,0]**2

iteration = 100
cost = np.zeros((iteration,1))
learning_rate = 0.00000001
#print(y)
#print(xNEW)
#plt.plot(x,y,'o')
#plt.show()
for i in range(iteration):
    r = np.dot(xNEW, theta) - y
    print(r)
    cost[i] = 0.5*np.sum(r*r)/xHeight
    theta = theta - learning_rate * np.dot(xNEW.T, r)
    print('step {}, cost: {}'.format(i, cost[i]))