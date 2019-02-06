#!/usr/bin/env python
# coding=utf-8

import sys
import os
#sys.path.append("..")
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from pyimage.models import MiniGoogleNet
from pyimage.callbacks import TrainingMonitor
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import argparse
from os.path import join

args = argparse.ArgumentParser()
args.add_argument("--output", required=True)
args.add_argument("--model", required=True)
args = args.parse_args()

BATCHSIZE=64
EPOCHS=70
LR=0.01

def poly_decay(epoch):
    maxEpochs = EPOCHS
    baseLR = LR
    power = 1.0
    alpha = baseLR * (1 - (epoch/float(maxEpochs))) ** power
    return alpha

print("[INFO] loading dataset...")
(trainX, trainY), (valX, valY) = cifar10.load_data()
trainX = trainX.astype("float32")
valX = valX.astype("float32")
mean = np.mean(trainX, axis=0)
print("mean.shape:", mean.shape)
trainX -= mean
valX -= mean
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
valY = lb.transform(valY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest", zoom_range=0.1)
figPath = join(args.output, "{}.png".format(os.getpid()))
jsonPath = join(args.output, "{}.json".format(os.getpid()))

callbacks = [TrainingMonitor(figPath, jsonPath), LearningRateScheduler(poly_decay)]

optimizer = SGD(lr=LR, momentum=0.9)
model = MiniGoogleNet.build(32, 32, 3, 10)
print("[INFO] compiling model...")
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCHSIZE), validation_data=(valX, valY), steps_per_epoch=len(trainX)//BATCHSIZE, epochs=EPOCHS, callbacks=callbacks)

print("[INFO] saving model...")
model.save(args.model)

