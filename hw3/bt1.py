import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')
height = data.shape[0]
width = data.shape[1]
x = data.iloc[:, 0]
x = x.values.reshape(-1, 1)
z = data.iloc[:, 1]
z = z.values.reshape(-1, 1)
y = data.iloc[:, 2]
y = y.values.reshape(-1, 1)

theta = np.zeros((width,1)).reshape(-1,1)
numiter = 1000
learningrate = 0.01
cost = np.zeros((numiter,1))
X = np.hstack((np.ones((height, 1)), x, z))

def sigmoi(Z):
    return 1/(1+np.exp(-Z))
def computecost(y,yHat):
    return -np.sum(y * np.log(yHat) + (1-y)*np.log(1-yHat))
def gradient(X,y,yHat):
    return np.dot(np.transpose(X),(yHat-y))

for i in range(1, numiter):
    loss = np.dot(X, theta)
    yHat = sigmoi(loss)
    cost = computecost(y,yHat)
    theta = theta - learningrate*gradient(X,y,yHat)
    print(cost)
    print(theta)

t = 0.5
plt.plot(x,yHat,'o')
plt.plot((4, 10),(-(theta[0]+4*theta[1]+ np.log(1/t-1))/theta[2], -(theta[0] + 10*theta[1]+ np.log(1/t-1))/theta[2]), 'g')
plt.show()
