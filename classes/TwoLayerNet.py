import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common.activation_function import *
from common.loss_function import *
from common.gradient import numeric_gradient
from common.layers import *
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self,input_size,hidden_size,output_size,weight_inti_std=0.01):

        self.params = {}
        self.params['W1'] = weight_inti_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_inti_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        #계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            y = layer.forward(x)
        '''
        W1, W2 = self.params['W1'], self.params['W2']
        b1 ,b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid_func(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = sigmoid_func(a2)
        y = softmax_func(z2)
        '''
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self,x,y):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        if t.ndim != 1:
            t = np.argmax(t,axis=1)

        _accuracy = np.sum(y == t) / float(x.shape[0])
        return _accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        
        loss_W = lambda W : self.loss(x,t)

        grads = {}
        grads['W1'] = numeric_gradient(loss_W,self.params['W1'])
        grads['b1'] = numeric_gradient(loss_W,self.params['b1'])
        grads['W2'] = numeric_gradient(loss_W,self.params['W2'])
        grads['b2'] = numeric_gradient(loss_W,self.params['b2'])

        return grads

    def gradient(self,x,t):
        #순전파
        self.loss(x,t)
        
        #역전파
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values()) #w1 b1 w2 b2
        layers.reverse() #b2 w2 b1 w1
        for layer in layers:
            dout = layer.backward(layer)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
