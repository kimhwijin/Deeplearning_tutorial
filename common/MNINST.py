import sys,os
from MNINST_dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from activation_function import sigmoid_func,softmax_func

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def init_MNINST():
    #load dataset
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)
    
    #test
    img = x_train[0]
    label = t_train[0]
    img = img.reshape(28,28)
    img_show(img)

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("MNINST_dataset/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

def predict(network,x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid_func(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid_func(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax_func(a3)

    return y


def test_predict():
    x,t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network,x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy :" + str(float(accuracy_cnt / len(x))))

def test_with_batch_predict():
    x,t = get_data()
    network = init_network()

    batch_size = 0
    accuracy_cnt = 0

    for i in range(0,len(x),batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network,x_batch)
        p = np.argmax(y_batch,axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy :" + str(float(accuracy_cnt / len(x))))
