import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv')
N = data.shape[0]
print(data)
x = data.iloc[:, 0]
x = x.values.reshape(-1, 1)
print(x)
y = data.iloc[:, 1]
y = y.values.reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('Diện tích')
plt.ylabel('Giá')
np.ones((N, 1))
x = np.hstack((np.ones((N, 1)), x))
w = np.array([0.,1.]).reshape(-1,1)
print(x)
print(w)
numOfIteration = 100
cost = np.zeros((numOfIteration,1))
learning_rate = 0.000001
for i in range(0, numOfIteration):
    r = np.dot(x, w) - y
    cost[i] = 0.5*np.sum(r*r)/N
    # correct the shape dimension
    w = w - learning_rate * np.dot(x.T, r)
    print('step {}, cost: {}'.format(i, cost[i]))

plt.figure(1)
plt.plot(x[:,1],y,'rx')
plt.show()
plt.figure(2)
plt.plot(cost)
plt.show()
