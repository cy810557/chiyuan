#!/usr/bin/env python
# coding=utf-8
from mxnet import nd
import numpy as np
from mxnet.gluon import nn
import mxnet as mx
from mxnet import gluon

class ConvBlock(nn.HybridBlock):
    def __init__(self, in_channels, channels, strides, padding, num_sync_bn_devices=-1, multiplier=1.0):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.HybridSequential()
        with self.conv_block.name_scope():
            self.conv_block.add(nn.Conv2D(int(channels*multiplier), 3, strides, padding,
                                          in_channels=in_channels, use_bias=False))
            if num_sync_bn_devices == -1:
                self.conv_block.add(nn.BatchNorm())
            else:
                self.conv_block.add(gluon.contrib.nn.SyncBatchNorm(num_devices=num_sync_bn_devices))
            self.conv_block.add(nn.Activation('relu'))
    def hybrid_forward(self, F, x):
        return self.conv_block(x)

class DepthwiseSeperable(nn.HybridBlock):
    def __init__(self, in_channels, channels, strides, num_sync_bn_devices=-1, multiplier=1.0, **kwags):
        # Weidth Multiplier
        in_channels = int(in_channels * multiplier)
        channels = int(channels * multiplier)
        super(DepthwiseSeperable, self).__init__(**kwags)
        self.depthwise = nn.HybridSequential()
        with self.depthwise.name_scope():
            self.depthwise.add(nn.Conv2D(in_channels, 3, strides, padding=1,groups=in_channels,
                                         in_channels=in_channels, use_bias=False))
            if num_sync_bn_devices == -1:
                self.depthwise.add(nn.BatchNorm())
            else:
                self.depthwise.add(gluon.contrib.nn.SyncBatchNorm(num_devices=num_sync_bn_devices))
            self.depthwise.add(nn.Activation('relu'))

        self.pointwise = nn.HybridSequential()
        with self.pointwise.name_scope():
            self.pointwise.add(nn.Conv2D(channels, 1, in_channels=in_channels, use_bias=False))
            if num_sync_bn_devices == -1:
                self.pointwise.add(nn.BatchNorm())
            else:
                self.pointwise.add(gluon.contrib.nn.SyncBatchNorm(num_devices=num_sync_bn_devices))
            self.pointwise.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return(self.pointwise(self.depthwise(x)))

class MobileNet(nn.HybridBlock):
    def __init__(self, num_classes, n_devices=2, multiplier=1.0, **kwags):
        super(MobileNet, self).__init__(**kwags)
        self.net = nn.HybridSequential()
        self.net.add(ConvBlock(3, 32, 2, 1, n_devices, multiplier))
        self.net.add(DepthwiseSeperable(32, 64, 1, n_devices, multiplier))
        self.net.add(DepthwiseSeperable(64, 128, 2, n_devices, multiplier))
        self.net.add(DepthwiseSeperable(128, 128, 1, n_devices, multiplier))
        self.net.add(DepthwiseSeperable(128, 256, 2, n_devices, multiplier))

        self.net.add(DepthwiseSeperable(256, 256, 1, n_devices, multiplier))
        self.net.add(DepthwiseSeperable(256, 512, 2, n_devices, multiplier))
        for _ in range(5):
            self.net.add(DepthwiseSeperable(512, 512, 1, n_devices, multiplier))
        self.net.add(DepthwiseSeperable(512, 1024, 2, n_devices, multiplier))
        self.net.add(DepthwiseSeperable(1024, 1024, 1, n_devices, multiplier))

        self.net.add(nn.GlobalAvgPool2D())
        self.net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        return self.net(x)
