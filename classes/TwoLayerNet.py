import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common.activation_function import *
from common.loss_function import *
from common.gradient import numeric_gradient

class TwoLayerNet:

    params = {}

    def __init__(self,input_size,hidden_size,output_size,weight_inti_std=0.01):
        self.params['W1'] = weight_inti_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_inti_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1 ,b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid_func(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = sigmoid_func(a2)
        y = softmax_func(z2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self,x,y):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
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
