import numpy as np
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    
    delta = 1e-7
    if y.ndim == 1:#데이터 1개
        t = t.reshape(1,t.size)
        y = y.reshape(1.y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta))/batch_size

import MNINST
def mini_batch(x_train,t_train,train_size,batch_size):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    return x_batch,t_batch


t = np.array([0,0,1,0,0,0,0,0,0,0])
y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
print(mean_squared_error(y,t))
print(cross_entropy_error(y,t))