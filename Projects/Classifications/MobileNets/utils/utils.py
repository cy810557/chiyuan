#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_log(logpath, viz=True, figName="training_curve.png"):
    df = pd.read_csv(logpath, delimiter='\t', index_col=False)
    plt.style.use('ggplot')
    df.columns = ["epoch", "train_acc","train_loss","val_acc","val_loss"]
    plt.plot(df['epoch'], df['train_loss'], label="train_loss")
    plt.plot(df['epoch'], df['train_acc'], label="train_acc")
    plt.plot(df['epoch'], df['val_loss'], label="val_loss")
    plt.plot(df['epoch'], df['val_acc'], label="val_acc")

    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    if viz: 
        plt.show() 
    else: 
        plt.savefig(figName)
