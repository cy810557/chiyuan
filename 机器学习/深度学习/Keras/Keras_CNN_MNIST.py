from keras.datasets import mnist
import keras
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D, Dropout
from keras import Sequential
import numpy as np
from keras import backend as K

img_rows, img_cols = 28, 28

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255
y_0 = [y_train, y_test]
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
#resize为送入卷积层提供4-D array形式
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

def CNN_model():
    model = Sequential()
    model.add(Conv2D(32,[3,3], activation='relu', input_shape=input_shape))
    model.add(Conv2D(64,[3,3],activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model = CNN_model()
model.fit(x_train, y_train,epochs=10, batch_size=128,verbose=1,validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print('=======================================')
print('Test loss :', score[0])
print('Test accuracy: ', score[1])