#%%
'''
y = h(b + x1w1 + x2w2)
h(x) : 0 (x <= 0) , 1 (x > 0)
'''
def sigmoid_func(x):
    hx = 1/(1 + np.exp(-x))
    return hx

import numpy as np
import matplotlib.pylab as plt
def step_func(x):
    y = x > 0
    return y.astype(np.int)

def relu_func(x):
    return np.maximum(0,x)

def identify_func(x):
    return x

def softmax_func(a):
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def multi_layer_sigmoid(network, X):
    W1 = network['W1']
    B1 = network['b1']
    A1 = np.dot(X,W1) + B1
    Z1 = sigmoid_func(A1)

    W2 = network['W2']
    B2 = network['b2']
    A2 = np.dot(Z1,W2) + B2
    Z2 = sigmoid_func(A2)

    W3 = network['W3']
    B3 = network['b3']
    A3 = np.dot(Z2,W3) + B3
    Y = A3
    return Y

network = init_network()
x = np.array([1.0,0.5])
y = multi_layer_sigmoid(network,x)
print(y)


x = np.arange(-5.0,5.0,0.1)
y = sigmoid_func(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


# %%
