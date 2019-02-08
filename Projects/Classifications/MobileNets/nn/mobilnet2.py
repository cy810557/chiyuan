# -*- coding: UTF-8 -*-

import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet.gluon import data as gdata
from mxnet.gluon.model_zoo import vision

class ReLU6(nn.HybridBlock):
    def __init__(self, **kwags):
        super(ReLU6, self).__init__(**kwags)
    
    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6)

def ConvBlock(channels, kernel_size, strides, padding=1, groups=1, activation='relu6'):
    block = nn.HybridSequential()
    block.add(nn.Conv2D(channels, kernel_size, strides, padding=padding, groups=groups, use_bias=False))
    block.add(nn.BatchNorm())
    if activation is not None:
        block.add(ReLU6(prefix='relu6_'))
    return block

def DepthWiseConv(channels, strides):
    return ConvBlock(channels, 3, strides, groups=channels)
    
def LinearBottleneck(channels):
    return ConvBlock(channels, 1, 1, 0, activation=None)

def ExpansionConv(channels):
    return ConvBlock(channels, 1, 1, 0)

class InvertedResidual(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, strides, t=6, **kwags):
        super(InvertedResidual,self).__init__(**kwags)
        self.strides = strides
        self.keep_channels = in_channels == out_channels
        expanded_channels = t * in_channels
        self.inver_residual = nn.HybridSequential()
        with self.inver_residual.name_scope():
            self.inver_residual.add(ExpansionConv(expanded_channels),
                                DepthWiseConv(expanded_channels, strides),
                                LinearBottleneck(out_channels))
    
    def hybrid_forward(self, F, x):
        out = self.inver_residual(x)
        if self.strides == 1 and self.keep_channels:
            out = out + x
            #out = F.elemwise_add(out, x)
        return out
    
def RepeatedInvertedResiduals(in_channels, out_channels, repeats, strides, t, **kwags):
    sequence = nn.HybridSequential(**kwags)
    # The first layer of each sequence has a stride s and all others use stride 1.
    sequence.add(InvertedResidual(in_channels, out_channels, strides, t))
    for _ in range(1, repeats):
        sequence.add(InvertedResidual(out_channels, out_channels, 1, t))
    return sequence

class MobileNetV2(nn.HybridBlock):
    def __init__(self, num_classes, width_multiplier=1.0, **kwags):
        super(MobileNetV2, self).__init__(**kwags)
        input_feature_channels = int(32 * width_multiplier)
        
        self.bottleneck_settings = [
            # t, c, n, s
            [1, 16, 1, 1, "stage0_"],      # -> 112x112
            [6, 24, 2, 2, "stage1_"],      # -> 56x56
            [6, 32, 3, 2, "stage2_"],      # -> 28x28
            [6, 64, 4, 2, "stage3_0_"],    # -> 14x14
            [6, 96, 3, 1, "stage3_1_"],    # -> 14x14
            [6, 160, 3, 2, "stage4_0_"],   # -> 7x7
            [6, 320, 1, 1, "stage4_1_"],   # -> 7x7
        ]
        self.net = nn.HybridSequential()
        self.net.add(ConvBlock(input_feature_channels, 3, 2))
        
        in_channels = input_feature_channels
        for t, c, n, s, prefix in self.bottleneck_settings:
            out_channels = int(width_multiplier * c)
            self.net.add(RepeatedInvertedResiduals(in_channels, out_channels, n, s, t, prefix=prefix))
            in_channels = out_channels  # 下一层的输入通道数为当前层的输出通道数
        
        # 注意：MobileNetV2使用的分类头不是GAP + Dense，而是GAP + 1x1 Linear Conv + Flatten
        self.net.add(ConvBlock(int(1280*width_multiplier), 1, 1, 0),
                          nn.GlobalAvgPool2D(),
                          nn.Conv2D(num_classes, 1, 1, 0, activation=None, use_bias=False),
                          nn.Flatten())
    
    def hybrid_forward(self, F, x):
        return self.net(x)