import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from classes.TwoLayerNet import *
from MNINST_dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

#hyper parameter
iters_num = 2000
batch_size = 100
learning_rate = 0.1
train_size = x_train.shape[0]

train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch,t_batch)

    #매개변수 갱신
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]
    
    #record iter loss
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    #accuracy per epoch 
    if i % iter_per_epoch == 0 and i != 0:
        train_accuracy = network.accuracy(x_train,t_train)
        test_accuracy = network.accuracy(x_test, t_test)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        print("train acc, test acc | " + str(train_accuracy) + ", " + str(test_accuracy))



