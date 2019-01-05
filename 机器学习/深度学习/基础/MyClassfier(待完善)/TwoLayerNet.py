import numpy as np
from layers import *
# 注意：在类中的函数相互调用时也需要加self，代表是类的函数。（但是在写参数的时候不需要显示地写self）


class TwoLayerNet():
    def __init__(self, input_dim=28*28,hidden_size=100, num_classes=10, learning_rate=1e-3, w_decay=1e-4, reg = 0.0):
        self.params = {}
        self.params['W1'] = w_decay * np.random.randn(input_dim, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = w_decay * np.random.randn(hidden_size, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        self.reg = reg
        self.lr = learning_rate

    def inference(self, X, y=None):
        grad = {}
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        #前向传播
        z1, cache_aff1 = affine_forward(X, W1, b1) #cache_af1记录第一层的输入，即第一层的local grad
        #另一种理解方法：使用BP：直接逆着算，把上游梯度看成输出，还用原来的W矩阵，乘积即作为输入神经元梯度
        a1, cache_relu = relu_forward(z1) #cache_relu记录第二层的输入，即第二层local grad
        scores, cache_scores = affine_forward(a1, W2, b2) #cache_scores记录第三层local_grad
        if y is None:
            return scores
        loss = None #python 中也需要先开辟变量空间吗？
        # 将scores 丢进 softmax_logistic_loss层，求出loss（前向传播的终点）以及dscores（反向传播的起点）
        loss ,dscores = softmax_logistic_loss(self, scores, y)
        d_a1, d_W2, d_b2 = affine_backward(self,  dscores, cache_scores)
        d_z1 = relu_backward(d_a1, cache_relu)
        d_X, d_W1, d_b1 = affine_backward(self, d_z1, cache_aff1)
        grad['W1'],grad['b1'] = W1, b1
        grad['W2'],grad['b2'] = W2, b2
        return loss, grad
    def predict(self, X):
        scores = self.inference(X)
        prediction = np.argmax(scores, axis=1)
        return prediction
    def train(self, X, y, lr_decay=0.98, reg=1e-4, num_iters=100, batch_size = 200):
        num_train = X.shape[0]
        iter_per_epoch = max(num_train/batch_size, 1)
        loss_history = []
        train_acc_history = []
        #注意：这段不熟练，需要多巩固
        for iter in range(num_iters):
            X_batch = None #注意要先清零
            y_batch = None
            indices = np.random.choice(num_train, batch_size, replace = True) #indices为选出的batch_size笔data
            X_batch = X[indices]
            y_batch = y[indices]
            loss, grad = self.inference(X_batch, y_batch)
            loss_history.append(loss)
            self.params['W1'] -= self.lr * grad['W1']
            self.params['b1'] -= self.lr * grad['b1']
            self.params['W2'] -= self.lr * grad['W2']
            self.params['b2'] -= self.lr * grad['b2']
            if iter%100 == 0:
                print('iteration {0}/{1}: loss {2}'.format(iter, num_iters, loss))
            if iter % iter_per_epoch==0:
                train_accuracy = (self.predict(X_batch) == y_batch).mean()
                train_acc_history.append(train_accuracy)
                self.lr *= lr_decay
        return {'loss history': loss_history,
                'train accuracy_history': train_acc_history}










        #loss, grad  = inference(self, X, y)






