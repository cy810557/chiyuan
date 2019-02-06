
# coding: utf-8

# In[42]:


import os
import re
import sys
import pdb
import time
import shutil
import random
import logging
import gluonbook as gb
from os.path import join
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, data as gdata
import matplotlib.pyplot as plt
import numpy as np


def set_transformer():
    transform_train = gdata.vision.transforms.Compose([
        # 将图像放大成宽高为40的正方形
        gdata.vision.transforms.Resize(40),
        # 随机对⾼和宽各为40像素的正⽅形图像裁剪出⾯积为原图像⾯积0.64 到1 倍之间的⼩正方
        # 形，再放缩为⾼和宽各为32像素的正⽅形。
        gdata.vision.transforms.RandomResizedCrop(32, scale=(0.64,1.0), ratio=(1.0,1.0)),
        # 随机左右翻转
        gdata.vision.transforms.RandomFlipLeftRight(),
        # 像素值归一化到0-1之间，并transpose channles
        gdata.vision.transforms.ToTensor(),
        # 对图像的每个通道做标准化
        gdata.vision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])])
    ## 注意：测试时无需对图像做标准化以外的数据增强处理。又因为验证集要和测试集
    ## 的处理保持一致，故同样
    transform_test = gdata.vision.transforms.Compose([
        # 注意转通道和标准化的顺序不要乱了
        gdata.vision.transforms.ToTensor(),
        gdata.vision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])   
    ])
    return transform_train, transform_test
    
def show_images(imgs, num_rows, num_cols, scale=2):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                               strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        
    def hybrid_forward(self, F, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk
    
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
def get_net(ctx):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=ctx, init=mx.init.Xavier())
    return net
        


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    acc = nd.array([0], ctx=ctx)
    n = 0
    for X, y in data_iter:
        X = X.as_in_context(ctx)
        y = y.astype('float32').as_in_context(ctx)
        acc += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc.asscalar() / n

def plot_history(train_loss_history, train_acc_history, val_acc_history):
    #plt.clf()
    ax_epoch = list(range(len(train_loss_history)))
    #pdb.set_trace()
    plt.subplot(1, 2, 1);
    plt.plot(ax_epoch, train_acc_history, 'r-');
    #plt.hold()
    plt.plot(ax_epoch, val_acc_history, 'g-.');
    plt.legend(labels=['train_acc','val_acc'], loc='upper right')
    plt.subplot(1, 2, 2);
    plt.plot(ax_epoch, train_loss_history, 'b-');
    plt.legend(labels = ['train loss'], loc = 'upper right')
    plt.show();
    plt.pause(0.00001)

def train(start_epoch, lr_decay_dict, ctx, num_epochs, val_iter, trainer, metric, save_name=None, log_name=None):
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    for epoch in range(start_epoch, num_epochs+1):

        tic = time.time()
        metric.reset()
        info_list = ''
        for iter, (X, y) in enumerate(train_iter):
            X = X.as_in_context(ctx)
            y = y.as_in_context(ctx)
            #y = nd.one_hot(y, 10).as_in_context(ctx)
            with autograd.record():
                logits = net(X)
                #pdb.set_trace()
                l = loss(logits, y)
                l.backward() 

            trainer.step(batch_size)
            metric.update([y,],[logits,])
            if iter % 100 == 0:
                name,train_acc = metric.get()
                info_iter = 'Epoch {}, Iter {}, train acc {}, train loss {}'.format(epoch, iter, train_acc, l.mean().asscalar())
                print(info_iter)
                info_list += info_iter+'\n'
        if epoch in lr_decay_dict.keys():
            for point in sorted(lr_decay_dict.keys()):
                if epoch > point:
                    trainer.set_learning_rate(trainer.learning_rate * lr_decay_dict[point])
            trainer.set_learning_rate(trainer.learning_rate * lr_decay_dict[epoch])  # lr schedule
            print('[+]Changing learning rate to {}'.format(trainer.learning_rate))
        if ((epoch % 50 == 0) and (epoch != start_epoch)): # or (epoch % 5 ==0 and epoch < 20)
            #trainer.set_learning_rate(trainer.learning_rate * lr_decay)
            net.save_parameters(save_name+'_epoch_{}'.format(epoch))
        name,train_acc = metric.get()
        valid_acc = evaluate_accuracy(val_iter, net, ctx)
        info_epoch =  '[+]Epoch {}, learning rate: {}, train accuracy: {}, train loss: {}, valid accuracy: {}, speed: {}'.format(
            epoch,  trainer.learning_rate, train_acc, l.mean().asscalar(), valid_acc, time.time()-tic)
        logging.info(info_epoch)
        info_list += info_epoch + '\n'
        if log_name is not None:
            with open(log_name,'a') as f:
                f.writelines(info_list)
        train_loss_history.append(l.mean().asscalar())
        train_acc_history.append(train_acc)
        val_acc_history.append(valid_acc)
        plot_history(train_loss_history, train_acc_history, val_acc_history)
    plt.savefig('logs/train_cifar10_res18_scheDict.png')
            
if __name__=='__main__':
    assert len(sys.argv)>1, "Usage: python Kaggle-Cifar10.py ckpt_name, *log_name.(Given len(sys.argv):{})".format(len(sys.argv))

    batch_size = 128
    ctx = mx.gpu(0)   
    lr = 0.0005
    wd = 5e-4  # 从5e-4 改成1e-4
    #lr_period = 75
    lr_decay_dict = {75:0.1, 150:0.1, 450:0.2, 650:0.6}
    num_epochs = 800
    ckpt = sys.argv[1]
    if len(sys.argv)>2:
        log_name = sys.argv[2]         #'logs/cifar_resnet18_schedual.log'
    else:
        log_name = None
    Train_Root = '../Dataset/Train/'
    Val_Root = '../Dataset/Valid/'
    Train_Val_Root = '../Dataset/Total/'
    plt.ion()
    train_ds = gdata.vision.ImageFolderDataset(Train_Root, flag=1)  #flag默认为1， 加载RGB图像
    valid_ds = gdata.vision.ImageFolderDataset(Val_Root, flag=1)
    total_ds = gdata.vision.ImageFolderDataset(Train_Val_Root, flag=1)
    transform_train, transform_test = set_transformer()
    train_iter = gdata.DataLoader(train_ds.transform_first(transform_train),
                             batch_size, shuffle=True, last_batch='keep')
    val_iter = gdata.DataLoader(valid_ds.transform_first(transform_test),
                               batch_size, shuffle=True, last_batch='keep')
    total_iter = gdata.DataLoader(total_ds.transform_first(transform_train),
                               batch_size, shuffle=True, last_batch='keep')
    #net = set_network(net='resnet', ckpt=ckpt)
    net = get_net(ctx)
    if ckpt is not None:
        print('[+]loading ckpt from {}'.format(ckpt))
        net.load_parameters(ckpt)
        #pdb.set_trace()
        start_epoch = int(re.findall(r'epoch\_(\d+)', ckpt)[0])
    else:
        start_epoch = 1
    print('[+]Start training from epoch {}'.format(start_epoch))
    net.hybridize()  #别忘记加 hybridize()

    trainer = gluon.Trainer(
       net.collect_params(), 'adam', 
       {'learning_rate' : lr, 'wd' : wd})
    # schedule = mx.lr_scheduler.MultiFactorScheduler(step=[75,150,450], factor=lr_decay)
    # schedule.base_lr = lr
    # adam_optimizer = mx.optimizer.Adam(wd=wd) #learning_rate=lr, lr_scheduler=schedule,
    # trainer = mx.gluon.Trainer(params=net.collect_params(), optimizer=adam_optimizer)
    #这里trainer先不设置学习率

    logging.basicConfig(level=logging.DEBUG)
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    
    logging.info('[+][+]Hyperparameters: lr: {}, wd: {}, lr_decay_dict: {}'.format(lr, wd, lr_decay_dict))
    logging.info('[+]Training Start...')
    
    train(start_epoch, lr_decay_dict, ctx, num_epochs, val_iter, trainer, metric,
    save_name='models_schedual/cifar_resnet18', log_name=log_name)
