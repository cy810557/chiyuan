import numpy as np
from TwoLayerNet import *
from gradient_check import *
import matplotlib.pyplot as plt
# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  np.random.seed(0)
  net = TwoLayerNet(input_size, hidden_size, num_classes, w_decay=1,learning_rate=1e-7)
  return net

def init_toy_data():
  np.random.seed(1)
  X = 10 * np.random.randn(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

net = init_toy_model()
X, y = init_toy_data()
# Forward pass: Compute loss
loss, _ = net.inference(X, y)
print(loss)

loss, grads = net.inference(X, y)

#定义一种相对误差函数用来评估梯度计算是否正确
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
# 运行gradient check
for param_name in grads:
  f = lambda W: net.inference(X, y)[0]
  param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
  print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

stats = net.train(X,y,num_iters=1000)
print('Final training loss: ', stats['loss history'][-1])

# plot the loss history
plt.plot(stats['loss  history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()