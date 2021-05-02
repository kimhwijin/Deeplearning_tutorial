#%%
'''
y = h(b + x1w1 + x2w2)
h(x) : 0 (x <= 0) , 1 (x > 0)
'''
import math
def sigmoid_func(x):
    hx = 1/(1 + math.exp(-x))
    return hx

import numpy as np
import matplotlib.pylab as plt
def step_func(x):
    y = x > 0
    return y.astype(np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_func(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


# %%
