import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from TwoLayerNet import *

# 准备数据。数据特征：1797个样本，每个样本包括8*8像素的图像和一个[0, 9]整数的标签
digits = load_digits()
digits = digits
digits_X = digits.data[0:1600]
digits_y = digits.target[0:1600]
X_train, X_test, y_train, y_test = train_test_split(digits_X, digits_y)

input = X_train.shape[1]
classes = len(set(y_train))
# 先随便选一些参数进行测试，看分类器是否能工作
net = TwoLayerNet(input_dim=input, hidden_size=200, num_classes=classes,
                  learning_rate=1e-3, w_decay=1e-2,reg = 1e-5)
net.train(X_train, y_train, num_iters=9000)
print(net.predict(X_test[1:10])==y_test[1:10])
