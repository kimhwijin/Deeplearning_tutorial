'''
perceptron
'''
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
    w1,w2,theta = 0,0,0.5
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

