import numpy as np

def _numeric_gradient_mysolution_with_batch(f,W):
    '''
    f :: loss function
    W :: parameters
    return :: gradients of each parameter
    '''
    h = 1e-4 #10**-4
    shape = W.shape
    W = W.reshape(W.size,)
    #파라미터 개수만큼
    grad = np.zeros_like(W)

    for idx in range(W.size):
        tmp_val = W[idx]

        #f(x + h)
        W[idx] = tmp_val + h
        fWh1 = f(W)

        #f(x - h)
        W[idx] = tmp_val - h
        fWh2 = f(W)

        grad[idx] = (fWh1 - fWh2) / (2*h)
        W[idx] = tmp_val
        
    W = W.reshape(shape)
    grad = grad.reshape(shape)
    return grad

def _numerical_gradient_1d(f,X):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad

def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

def numeric_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) #f(x+h)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x) #f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        it.iternext()
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numeric_gradient(f,x)
        x -= lr * grad
    return x
