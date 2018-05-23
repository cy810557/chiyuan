import numpy as np
from TwoLayerNet import *
def affine_forward(x, W, b):
    #注意：由于输入的形式不确定，可能是3-D的，需要先reshape，再乘以weight
    # 要reshape成 N X D的形式，就要先知道N
    #
    output = None
    N = x.shape[0]
    X = np.reshape(x,(N, -1))
    output = np.dot(X, W) + b

    #将要保存的本地梯度信息存放在cache中，形式为一个tuple
    #注意：！！！！cache中保存的 x是小写的，即reshape之前的
    cache = x, W, b
    return output, cache

def affine_backward(self, dout, cache):
    '''
    dout: upstream gradient with respect to neurons. dim:n X N_l+1
    cache:
        X: local gradient of W.  dim: n X N_l
        W: local gradient of X.  dim: N_l X N_l+1
        b:                       dim: N_l+1
    '''
    # 注意：cache中的x不一定是可以直接乘以W的形式
    x, W, b = cache
    N = x.shape[0]
    X = np.reshape(x, (N,-1))
    dx = np.dot(dout, W.T).reshape(x.shape) #问题：dx不需要惩罚吗
    dW = np.dot(X.T, dout) + self.reg * W
    db = np.sum(dout, axis=1) #暂时不理解为什么是这样写

    return dx, dW, db

def relu_forward(X):
    output = np.maximum(X,0)
    cache = X
    return output, cache
def relu_backward(dout, cache):
    X  = cache
    d_x = dout * (X>0)
    return d_x
def softmax_logistic_loss(self, scores, y):
    W1, W2 = self.params['W1'], self.params['W2']
    # 先将score转换概率
    probs = np.exp(scores - np.max(scores,axis=1, keepdims=True))
    probs/=np.sum(probs,axis=1,keepdims=True)
    #减去最大值避免指数爆炸

    N =scores.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N),y])) + 0.5 * self.reg* (np.sum(W1*W1) + np.sum(W2*W2))
    loss /= N
    dx = probs.copy() #使用copy强制不关联（赋值时要注意的）
    dx[np.arange(N),y]-=1
    dx/=N
    return loss, dx



