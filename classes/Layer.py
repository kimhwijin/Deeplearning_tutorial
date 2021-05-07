import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from common.loss_function import *
from common.activation_function import *

#곱셈 노드
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx ,dy

#덧셈 노드
class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        '''type(x) = numpy'''
        self.mask = (x <= 0) #0이하면 True, 이외 False
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0 #x>0 : dL/dy, x <= 0 : 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):
        dx = dout * (1.0 - dout) * self.out  #dL/dy * y * (1-y)
        return dx

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,W) + self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T) #dL/dx = dL/dy * W.T
        self.dW = np.dot(self.x.T,dout) #dL/dW = x.T * dL/dy
        self.db = np.sum(dout,axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax_func(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  #y - t
        return dx
