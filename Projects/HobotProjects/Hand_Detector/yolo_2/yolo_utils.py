# -*- coding:utf-8 -*-
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon,nd
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import Block, HybridBlock
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import pdb
import time 


RGB_MEAN = nd.array([123, 117, 104])
RGB_STD = nd.array([58.395, 57.12, 57.375])
anchor_scales = scales = [[4.0, 4.8],   #每一行表示一个预先选定的bbox形状，第一列为width，第二列为height。【【这里的数值不是在全图上的数值，而是根据网络输出feature map的大小(256/16=16)定的】】
                  [4.8, 4.8],
                  [5.6, 4.0],
                  [4.8, 8.0],
                  [6.4, 6.4]]
				  
#这两个helper function不是自己写的，需要日后详细研究
def transform_center(xy):
    """Given x, y prediction after sigmoid(), convert to relative coordinates (0, 1) on image."""
    b, h, w, n, s = xy.shape
    offset_y = nd.tile(nd.arange(0, h, repeat=(w * n * 1), ctx=xy.context).reshape((1, h, w, n, 1)), (b, 1, 1, 1, 1))
    # print(offset_y[0].asnumpy()[:, :, 0, 0])
    offset_x = nd.tile(nd.arange(0, w, repeat=(n * 1), ctx=xy.context).reshape((1, 1, w, n, 1)), (b, h, 1, 1, 1))
    # print(offset_x[0].asnumpy()[:, :, 0, 0])
    x, y = xy.split(num_outputs=2, axis=-1)
    x = (x + offset_x) / w
    y = (y + offset_y) / h
    return x, y
def transform_size(wh, anchors):
    """Given w, h prediction after exp() and anchor sizes, convert to relative width/height (0, 1) on image"""
    b, h, w, n, s = wh.shape
    aw, ah = nd.tile(nd.array(anchors, ctx=wh.context).reshape((1, 1, 1, -1, 2)), (b, h, w, 1, 1)).split(num_outputs=2, axis=-1)
    w_pred, h_pred = nd.exp(wh).split(num_outputs=2, axis=-1)
    w_out = w_pred * aw / w
    h_out = h_pred * ah / h
    return w_out, h_out
    
def yolo2_feature_spliter(feature, num_classes, anchor_scales):
    '''
    Transpose/Reshape/Organize convolution outputs.
    '''
    stride = num_classes + 5
    feature = nd.transpose(feature,[0,2,3,1])  #(32,16,16,14)
    feature = feature.reshape((0,0,0,-1,stride))   #(32,16,16,2,7)
    # class probs
    cls_pred = feature.slice_axis(begin=0, end=num_classes, axis=-1)
    # object score
    score_pred = feature.slice_axis(begin=num_classes, end=num_classes+1, axis=-1)
    scores = nd.sigmoid(score_pred)
    # center prediction, in range(0,1) for each grid
    xy_pred = feature.slice_axis(begin=num_classes+1, end=num_classes+3, axis=-1)
    xy = nd.sigmoid(xy_pred)
    #pdb.set_trace()
    # 注意：此时的每个grid的中心坐标(x,y)表示的是位于当前grid cell的相对位置, 在最后预测阶段使用的是相对于全图的位置
    x, y = transform_center(xy)
    
    # width/height prediction
    wh = feature.slice_axis(begin=num_classes+3, end=num_classes+5, axis=-1)
    # 同理，在后面的预测阶段需要将长度和宽度转换为相对于全图的长、宽
    #pdb.set_trace()
    w, h = transform_size(wh, anchor_scales)
    
    # final class prediction
    category = nd.argmax(cls_pred, axis=-1, keepdims=True)
    
    # 注意：训练阶段使用的是【中心+长宽】的bbox，而最终预测阶段使用的思【左上角+右下角】的bbox，故提前准备好预测使用的bbox(都是相对全图的坐标)
    # 注意：一个细节：某些预测bbox的中心坐标可能位于图像边缘，且长宽已超出边界。这样当转换为corner坐标会出现负的或大于1.
    left = nd.clip(x-w/2, 0, 1)
    top = nd.clip(y-h/2, 0, 1)
    right = nd.clip(x+w/2, 0, 1)
    bottom = nd.clip(y+h/2, 0, 1)
    
    output_to_draw = nd.concat(*[category, scores, left, top, right, bottom], dim=-1)
    # 注意：这里必须加星号。否则 mxnet AssertionError: Positional arguments must have NDArray type, but got [...
    return output_to_draw, cls_pred, scores, nd.concat(*[xy, wh], dim=-1)
    
class Yolo2_Output(HybridBlock):
    def __init__(self, num_classes, anchor_scales):
        super(Yolo2_Output, self).__init__()
        assert len(anchor_scales)>0, "at least one anchor scale required"
        assert num_classes>0, "num of class should >0, given{}".format(num_classes)
        self._anchor_scales = anchor_scales
        output_channel = len(anchor_scales) * (num_classes + 1 + 4)
        with self.name_scope():
            self.output = nn.Conv2D(output_channel,1,1)
    def hybrid_forward(self,F,x):
        return self.output(x)
def creat_model(params_path, num_classes, anchor_scales):
    pretrained = vision.get_model('resnet18_v1').features
    net = nn.HybridSequential()
    for i in range(len(pretrained)-2):
        net.add(pretrained[i])
    predictor = Yolo2_Output(num_classes=num_classes, anchor_scales=anchor_scales)
    net.add(predictor)
    net.load_parameters(params_path)
    return net
    
    
def process_image(frame):
    im = nd.array(frame[...,::-1])
    # resize to data_shape
    data = mx.image.imresize(im, 256, 256)
    # minus rgb mean, divide std
    data = (data.astype('float32') - RGB_MEAN) / RGB_STD
    # convert to batch x channel x height xwidth
    return data.transpose((2,0,1)).expand_dims(axis=0)

def predict(net, x):
    t1 = time.time()
    x = net(x)
    #print('time for one forward pass: ', (time.time()-t1)*1000, 'ms')
    output, cls_prob, score, xywh = yolo2_feature_spliter(x, 2, anchor_scales)
    return nd.contrib.box_nms(output.reshape((0, -1, 6)))
    
def display2(frame, out, save_name=None):
    box = out[0][0][2:6] * np.array([frame.shape[1],frame.shape[0]]*2)
    cv2.rectangle(frame, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 4)
    cv2.imshow('test_frame', frame)
    #cv2.imwrite('test_obj.jpg', frame)
def evaluate(net, n_epoch, n_iter):
    test_batch  = test_data.next()
    test_img = test_batch.data[0].as_in_context(ctx)
    test_label = test_batch.label[0].as_in_context(ctx)
    test_out = predict(net, test_img)
    test_pred = test_out[:,0,-4:]
    test_lb = test_label.squeeze()[:,1:]
    loss = nd.mean(l1_loss(test_pred, test_lb))
    print('Epoch: {0}    Iteration: {1}    Evaluation loss: {2}'.format(n_epoch, n_iter, loss))
    plot_prediction(test_img, test_pred, n_epoch, n_iter)
    return loss