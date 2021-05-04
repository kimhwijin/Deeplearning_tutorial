import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.loss_function import cross_entropy_error
from common.activation_function import softmax_func
from common.gradient import numeric_gradient


#2 x 3의 형상인 가중치 매개변수를 가진 모델
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    #예측 수행(함수값 리턴)
    def predict(self,x):
        return np.dot(x,self.W)

    #x = 입력값, t = 정답
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax_func(z)
        loss = cross_entropy_error(y,t)

        return loss