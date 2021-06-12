import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv')
x = data.values[:,0]
y = data.values[:,1]
plt.plot(x,y)
plt.show()

