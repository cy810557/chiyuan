from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from TwoLayerNet import *

digits = load_digits()
digits_X = digits.data
digits_y = digits.target
X_train, X_test, y_train, y_test = train_test_split(digits_X, digits_y)
input = X_train.shape[1]
classes = len(set(y_train))

my_net = TwoLayerNet(input_dim=input, hidden_size=50, num_classes=classes,
                  learning_rate=1e-3, w_decay=1,reg = 1e-5)
my_net.fit(X_train, y_train)
# 不行，因为自己写的分类器没有.fit方法
print(my_net.predict(X_test[1:10]))

