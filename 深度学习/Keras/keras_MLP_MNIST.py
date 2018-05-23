import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

# model = Sequential()
# model.add(Dense(input_dim=100, units=64, activation='relu'))
# model.add(Dense(units=10, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# data = np.random.random((1000, 100))
# label = np.random.randint(10, size = (1000, 1))
# one_hot_label = keras.utils.to_categorical(label, num_classes=10)
# model.fit(data, one_hot_label, epochs=10, batch_size=32)
## 使用keras训练一个分类器用于mnist
from keras.datasets import mnist

batch_size = 128
num_classes = 10
epochs = 5

(x_train, y_train),(x_test, y_test) = mnist.load_data() #注意：加载出来的为ndarry形式(None, 28, 28),type:uint8
Y0 = np.r_[y_train, y_test]
x_train = x_train.reshape(-1, 784).astype('float32')/255
x_test = x_test.reshape(-1, 784).astype('float32')/255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train,num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
X = np.r_[x_train, x_test]
Y = np.r_[y_train, y_test]
def create_model():
    model = Sequential()
    model.add(Dense(units=512, input_dim=784, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss: ', score[0])
# print('Test accuracy: ', score[1])

# 将定义好的model传入sklearn进行cross_validation
model = KerasClassifier(build_fn=create_model, nb_epoch = 20, batch_size = 128)
kfold = StratifiedKFold(n_splits=3, shuffle = True, random_state=41)
# results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
scores_history = []
for train_ind, test_ind in kfold.split(X, Y0):
    model = Sequential()
    model.add(Dense(units=512, input_dim=784, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X[train_ind], Y[train_ind], epochs = epochs, batch_size = batch_size, verbose= 0)
    scores = model.evaluate(X[test_ind], Y[test_ind], verbose=0)
    scores_history.append(scores)



