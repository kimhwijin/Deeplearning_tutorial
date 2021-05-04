import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from classes.simpleNet import *

net = simpleNet()
print(net.W)
x = np.array([0.6,0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0,0,1])
print(net.loss(x,t))

f = lambda w: net.loss(x,t)

dW = numeric_gradient(f,net.W)
print(dW)