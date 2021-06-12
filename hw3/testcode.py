import numpy as np
a = 5
b = 3
x = [-2.5,2.5]
y = np.sqrt((b**2)*(1-(np.power(x,2)/a**2)))
print(y)