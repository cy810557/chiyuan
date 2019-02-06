#!/usr/bin/env python
# coding=utf-8
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD
from os.path import join
import os
from datetime import datetime
import sys
from imutils import paths
import numpy as np
sys.path.append("..")
from pyimage.utils import SimpleDatasetLoader, plot_history, rank5_accuracy
from pyimage.preprocess import *
from pyimage.callbacks import TrainingMonitor
from keras.applications import resnet50, inception_resnet_v2
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.models import Model, Input
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
from keras.utils import to_categorical
from pyimage.utils import plot_history


TRAINROOT = "../Dataset/Seeds/train/"
SAVEWEIGHTS='outputs/models/baseline_incepRes_noaug'
OUTPUTPATH = 'outputs/logs/'
BATCHSIZE=32
LR=0.001
SIZEINPUT=299
INPUTSHAPE=(299,299,3)
POOLING = 'avg'
def load_dataset(k=7):

    def _read_img(filepath, size):
        img = load_img(filepath, target_size=size)
        img = img_to_array(img)
        return img

    imgPaths = list(paths.list_images(TRAINROOT))
    np.random.shuffle(imgPaths)
    X = np.zeros((len(imgPaths), SIZEINPUT, SIZEINPUT, 3), dtype='float32')
    processbar = tqdm(imgPaths)
    for i, imgPath in enumerate(processbar):
        img = _read_img(imgPath, (SIZEINPUT, SIZEINPUT))
        x = inception_resnet_v2.preprocess_input(np.expand_dims(img, axis=0))
        X[i] = x  # 这里其实x和X[i]维度不一样
        processbar.update()
    #X = inception_resnet_v2.preprocess_input(X) #对整个数据集的X用preprocess结果不一样...
    y = [x.split(os.path.sep)[-2] for x in imgPaths]
    print('Train Images shape: {} size: {:,}'.format(X.shape, X.size))
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X, y))
    return folds, X, y

def get_model_v2():
    baseModel = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                input_tensor=Input(shape=(SIZEINPUT,SIZEINPUT,3)))
    for layer in baseModel.layers:
        layer.trainable=False

    features = baseModel.output
    x = GlobalAveragePooling2D(name="avg_pool")(features)
    #x = Flatten()(x) #注意：使用GlobalAveragePooling2D时，输出即为2D Tensor，无需再用Flatten
    x = Dense(12, activation='softmax')(x)
    x= Dropout(0.5)(x)
    model = Model(inputs=baseModel.inputs, outputs=x)

    sgd = SGD(lr=LR, momentum=0.9)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

folds, X, y = load_dataset()
lb = LabelBinarizer()
y = lb.fit_transform(y)
classNames = lb.classes_
y = y.astype('float32')

def get_one_fold(X, y, num_fold=0):
    if y.shape[-1]!=12:
        y = to_categorical(y, 12)
    train_idx, val_idx = folds[num_fold]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[val_idx]
    y_valid = y[val_idx]
    return X_train, y_train, X_valid, y_valid

print("[INFO] loading fold-1 Dataset...")
X_train, y_train, X_valid, y_valid = get_one_fold(X, y)
aug = ImageDataGenerator(rotation_range=360, 
                         width_shift_range=0.1, 
                         height_shift_range=0.1,
                         #brightness_range=[0.5, 1.5],
                         zoom_range=0.2,
                         shear_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                        )
generator = aug.flow(X_train, y_train, batch_size=BATCHSIZE)


print("[INFO] building model...")
model = get_model_v2()
'''
model.load_weights('21_32.hdf5')
for layer in model.layers[607:]:
    layer.trainable = True
model.load_weights('unfreeze_conv8x8.h5')
opt = SGD(lr=0.001, momentum=0.9)
model.compile(opt, "categorical_crossentropy", metrics=["accuracy"])
print('[INFO] continue training 8x8 conv layers...')
'''
print('[INFO] start training...')
epochs=20
lrscheduler = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
ckpt = ModelCheckpoint("unfreeze_fc.h5", save_best_only=True, monitor='val_loss', mode='min', period=1)
callbacks = [lrscheduler, ckpt]
H = model.fit_generator(generator, validation_data=(X_valid, y_valid), 
                    steps_per_epoch=len(X_train)//BATCHSIZE, epochs=epochs, callbacks=callbacks)
print("[INFO] evaluating...")
plot_history(H, epochs, "unfreeze_fc_warmstart.png")
#plot_history(H, 50, "unfreeze_to_block_17_5.png")
preds = model.predict(X_valid)
print(classification_report(y_valid.argmax(axis=1),preds.argmax(axis=1)))
print("[INFO] saving final models...")
#model.save_weights('unfreeze_conv8x8.h5')
model.save_weights('unfreeze_bottom_fc.h5')

