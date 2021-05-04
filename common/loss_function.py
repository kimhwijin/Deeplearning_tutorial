import numpy as np
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error_one_hot_label(y,t):

    delta = 1e-7
    if y.ndim == 1:#데이터 1개
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta))/batch_size

def cross_entropy_error(y,t):
    '''
    y :: np array 형식으로 계산을 통해 출력된 레이블
    t :: np array 형식으로 입력값에 따른 정확한 레이블
    return :: 
    -1 * sum of (t ln y) / batch_size 의 계산식
    정답 레이블과 출력 레이블의 오차를 log 함수를 통해 반환함
    '''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


