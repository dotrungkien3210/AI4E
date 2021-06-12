import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = 2
fx = x**2
derifx = 2*x
learning_rate = 0.001
epoch = 5000
listx = []
listfx = []
def gradient(x,fx):
    for i in range (0,epoch):
        listx.append(x)
        listfx.append(fx)
        x = x - learning_rate*derifx
        fx = x**2
        if round(fx,1) == 0:
            print('Reach optima at I = % d; J = %f' % (i, fx))
            break
gradient(x,fx)
print(listx)
print(listfx)
plt.plot(listx,listfx)
plt.show()





