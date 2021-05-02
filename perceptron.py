'''
perceptron
'''
#w1x1 + w2x2 <= theta or > theta
def AND(x1, x2):
    w1, w2,theta = 0.5,0.5,0.7
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

def NAND(x1,x2):
    w1,w2,theta = -0.5,-0.5,-0.7
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

def OR(x1,x2):
    w1,w2,theta = 0.5,0.5,0.2
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

#b + w1x1 + w2x2 <= 0 or > 0
import numpy as np
def bias_AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    temp = b + np.sum(x*w)
    if temp <= 0:
        return 0
    elif temp > 0:
        return 1

def bias_NAND(x1,x2):
    x = np.arran([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    temp = b + np.sum(x*w)
    if temp <= 0:
        return 0
    elif temp > 0:
        return 1

def bias_OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    temp = b + np.sum(x*w)
    if temp <= 0:
        return 0
    elif temp > 0:
        return 1

def XOR(x1,x2):
    return "fail"