# -*- coding:utf-8 -*-
# 加入修改：①所有卷积层之后加入BN②除了substage,各个stage最后一层conv 去除activation
import mxnet as mx
import mxnet.gluon as g
from mxnet.gluon import nn


def conv_2d(sub_model, channels, kernel_size, activation='relu'):
    '''
    定义一个用于实现same conv的helper function
    activation: 'relu' or None

    '''
    sub_model.add(nn.Conv2D(channels=channels, kernel_size=kernel_size,
                            strides=1, padding=int((kernel_size - 1) / 2), activation=None))
    sub_model.add(nn.BatchNorm()) # beta_initializer='zeros', gamma_initializer='ones'
    if activation is not None:
        sub_model.add(nn.Activation(activation))


def max_pooling(sub_model):
    sub_model.add(nn.MaxPool2D(strides=2))


class CPM(g.nn.HybridBlock):
    def __init__(self, stages, joints):
        super(CPM, self).__init__()

        self.stages = stages
        self.stage_heatmap = []
        self.joints = joints
        model_stages = []

        with self.name_scope():
            # 模型一：feature extractor
            self.sub_stage = nn.HybridSequential('sub_stage')
            with self.sub_stage.name_scope():
                conv_2d(self.sub_stage, channels=64, kernel_size=3)
                conv_2d(self.sub_stage, channels=64, kernel_size=3)
                max_pooling(self.sub_stage)
                conv_2d(self.sub_stage, channels=128, kernel_size=3)
                conv_2d(self.sub_stage, channels=128, kernel_size=3)
                max_pooling(self.sub_stage)

                conv_2d(self.sub_stage, channels=256, kernel_size=3)
                conv_2d(self.sub_stage, channels=256, kernel_size=3)
                conv_2d(self.sub_stage, channels=256, kernel_size=3)
                conv_2d(self.sub_stage, channels=256, kernel_size=3)
                max_pooling(self.sub_stage)

                conv_2d(self.sub_stage, channels=512, kernel_size=3)
                conv_2d(self.sub_stage, channels=512, kernel_size=3)
                conv_2d(self.sub_stage, channels=512, kernel_size=3)
                conv_2d(self.sub_stage, channels=512, kernel_size=3)
                conv_2d(self.sub_stage, channels=512, kernel_size=3)
                conv_2d(self.sub_stage, channels=512, kernel_size=3)

                conv_2d(self.sub_stage, channels=128, kernel_size=3)

            self.model_stage_1 = nn.HybridSequential('model_stage_1')
            with self.model_stage_1.name_scope():
                conv_2d(self.model_stage_1, channels=512, kernel_size=1)
                conv_2d(self.model_stage_1, channels=self.joints + 1, kernel_size=1,activation=None)

            self.model_stage_2 = nn.HybridSequential('stage2_')
            with self.model_stage_2.name_scope():
                # input
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=1)
                conv_2d(self.model_stage_2, channels=self.joints + 1, kernel_size=1,activation=None)
                # output append

            self.model_stage_3 = nn.HybridSequential('stage3_')
            with self.model_stage_3.name_scope():
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=1)
                conv_2d(self.model_stage_3, channels=self.joints + 1, kernel_size=1,activation=None)

            self.model_stage_4 = nn.HybridSequential('stage4_')
            with self.model_stage_4.name_scope():
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=1)
                conv_2d(self.model_stage_4, channels=self.joints + 1, kernel_size=1,activation=None)

            self.model_stage_5 = nn.HybridSequential('stage5_')
            with self.model_stage_5.name_scope():
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=1)
                conv_2d(self.model_stage_5, channels=self.joints + 1, kernel_size=1,activation=None)

            self.model_stage_6 = nn.HybridSequential('stage6_')
            with self.model_stage_6.name_scope():
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=1)
                conv_2d(self.model_stage_6, channels=self.joints + 1, kernel_size=1,activation=None)

    def forward(self, x):
        self.sub_stage_img_feature = self.sub_stage(x)
        self.stage1_heatmap = self.model_stage_1(self.sub_stage_img_feature)
        self.stage1_featuremap = mx.ndarray.concat(self.stage1_heatmap, self.sub_stage_img_feature)
        self.stage2_heatmap = self.model_stage_2(self.stage1_featuremap)
        self.stage2_featuremap = mx.ndarray.concat(self.stage2_heatmap, self.sub_stage_img_feature)
        self.stage3_heatmap = self.model_stage_3(self.stage2_featuremap)
        self.stage3_featuremap = mx.ndarray.concat(self.stage3_heatmap, self.sub_stage_img_feature)
        self.stage4_heatmap = self.model_stage_4(self.stage3_featuremap)
        self.stage4_featuremap = mx.ndarray.concat(self.stage4_heatmap, self.sub_stage_img_feature)
        self.stage5_heatmap = self.model_stage_5(self.stage4_featuremap)
        self.stage5_featuremap = mx.ndarray.concat(self.stage5_heatmap, self.sub_stage_img_feature)
        self.stage6_heatmap = self.model_stage_6(self.stage5_featuremap)
        self.totol_heatmap = mx.ndarray.concat(self.stage1_heatmap, self.stage2_heatmap, self.stage3_heatmap,
                                               self.stage4_heatmap, self.stage5_heatmap, self.stage6_heatmap)
        return self.totol_heatmap  # label contact 6次 L2得到 loss