import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = 5
b = 3
x = np.arange(-5,5.1,0.1)
y = np.sqrt((b**2)*(1-(np.power(x,2)/a**2)))
z = -y
X = np.array([[0,3],[5,0],[-5,0],[0,-3],[-2.5,2.6],[2.5,2.6],[2.5,-2.6],[-2.5,-2.6]])
Y = [1,1,1,1,0,0,0,0]
height = X.shape[0]
width = X.shape[1]
m = X[:,0]
n = X[:,1]
X = np.hstack((np.ones((height, 1)), m, n))
theta = np.zeros((width,1))
print(X)
print(theta)






