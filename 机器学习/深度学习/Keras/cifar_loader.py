from keras.datasets import cifar10
import keras
import keras.backend as K
import cv2
import numpy as np
nb_train_sample = 3000 #假设在训练集不大的情况下使用预训练模型
nb_val_sample = 100
nb_test_sample = 100
def load_cifar10_keras(img_row, img_col):
    (X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
    x_train = [cv2.resize(img, (img_row, img_col)) for img in X_train[:nb_train_sample,:,:,:]]
    x_val = [cv2.resize(img, (img_row,img_col)) for img in X_val[:nb_val_sample,:,:,:]]
    x_test = [cv2.resize(img, (img_row, img_col)) for img in X_val[nb_val_sample:nb_test_sample+nb_val_sample,:,:,:]]
    y_train = keras.utils.to_categorical(Y_train[:nb_train_sample])
    y_val = keras.utils.to_categorical(Y_val[:nb_val_sample])
    y_test = keras.utils.to_categorical(Y_val[nb_val_sample:nb_test_sample+nb_val_sample])
    return np.array(x_train),np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)
