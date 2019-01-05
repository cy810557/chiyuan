# -*- coding:utf-8 -*-

import mxnet as mx
import matplotlib.pyplot as plt
from mxnet import nd
from yolo_v2 import yolo2_feature_spliter, yolo2_target, transform_size, transform_center
import numpy as np
from mxnet import gluon

l1_loss = gluon.loss.L1Loss()

RGB_MEAN = nd.array([123, 117, 104])
RGB_STD = nd.array([58.395, 57.12, 57.375])
#anchor_scales = [[3.2,5.6],[4,4],[5.6,3.2]]
anchor_scales = [[4.0, 4.8],   #每一行表示一个预先选定的bbox形状，第一列为width，第二列为height。【【这里的数值不是在全图上的数值，而是根据网络输出feature map的大小(256/16=16)定的】】
          [4.8, 4.8],
          [5.6, 4.0],
          [4.8, 8.0],
          [6.4, 6.4]]
class LossRecorder(mx.metric.EvalMetric):
    def __init__(self, name):
        super(LossRecorder, self).__init__(name)
    def update(self, labels, preds=0):
        for loss in labels:
            if isinstance(loss, mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1

def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)

def viz_labels_with_bbox(images, labels):
    ctx = images.context
    _, figs = plt.subplots(3, 3, figsize=(10,10))
    for i in range(3):
        for j in range(3):
            img, label = images[3*i+j], labels[3*i+j][0]
            img = img.transpose((1, 2, 0)) * RGB_STD.as_in_context(ctx) + RGB_MEAN.as_in_context(ctx)
            img = img.clip(0,255).asnumpy()/255
            fig = figs[i][j]
            fig.imshow(img)
            #for label in labels:
            rect = box_to_rect(label[1:5] * 256,'red',2)
            fig.add_patch(rect)

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
    plt.show()

def _predict(img, net):
    rgb_mean = RGB_MEAN.as_in_context(img.context)
    rgb_std = RGB_STD.as_in_context(img.context)
    assert len(img.shape)==3, "test image shape invalid. expected 3 dims, given shape: {}".format(img.shape)
    i0 = img.transpose((1,2,0))
    img = img.expand_dims(axis=0)
    feature = net(img)
    output, cls_prob, score, xywh = yolo2_feature_spliter(feature, 2, anchor_scales)
    # nms
    out = nd.contrib.box_nms(output.reshape((0, -1, 6)))
    out = out.asnumpy()

    box = out[0][0][2:6] * np.array([img.shape[-2], img.shape[-1]] * 2)
    rect = box_to_rect(nd.array(box), 'green', 2)
    i0 = (i0 * rgb_std + rgb_mean).asnumpy()
    i0 = i0.clip(0, 255)/255.
    plt.imshow(i0)
    plt.gca().add_patch(rect)
    try:
        plt.show()
    except:
        print('plt.show Connection refused...')
        plt.savefig(save_name)
    return box
def plot_prediction(test_img, test_pred, n_epoch, n_iter):
    _, figs = plt.subplots(3, 3, figsize=(12,12))
    for i in range(3):
        for j in range(3):
            im, l = test_img[3*i+j], test_pred[3*i+j]
            im = im.transpose((1, 2, 0)) * RGB_STD + RGB_MEAN
            im = im.clip(0,255).asnumpy()/255
            fig = figs[i][j]
            fig.imshow(im)
            rect = box_to_rect(l * data_shape,'red',2)
            fig.add_patch(rect)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
    
    plt.savefig('training_eval_continue_anchor_5/{0}_{1}_eval.jpg'.format(n_epoch,n_iter))
    #plt.show()
    plt.clf()	
def predict(net, x):
    x = net(x)
    #print('time for one forward pass: ', (time.time()-t1)*1000, 'ms')
    output, cls_prob, score, xywh = yolo2_feature_spliter(x, 2, anchor_scales)
    return nd.contrib.box_nms(output.reshape((0, -1, 6)))
	
def check_tbox(image, label):
    plt.clf()
    rgb_mean = RGB_MEAN.as_in_context(image.context)
    rgb_std = RGB_STD.as_in_context(image.context)
    assert label.shape == (1, 5), \
        "shape of label expected [1, 5], but given {}".format(label.shape)
    assert image.shape == (3, 256, 256), \
        "shape of image expected [3, 256, 256], given {}".format(image.shape)
    scores_tmp = nd.zeros((1, 16, 16, 3, 1))
    label = label.expand_dims(axis=0)
    tid, tscore, tbox, _ = yolo2_target(scores_tmp, label, anchor_scales)
    t_xy = tbox.slice_axis(begin=0, end=2, axis=-1)
    t_wh = tbox.slice_axis(begin=2, end=4, axis=-1)
    xy = nd.sigmoid(t_xy)
    x, y = transform_center(xy)
    w, h = transform_size(t_wh, anchor_scales)

    left = nd.clip(x - w / 2, 0, 1)
    top = nd.clip(y - h / 2, 0, 1)
    right = nd.clip(x + w / 2, 0, 1)
    bottom = nd.clip(y + h / 2, 0, 1)

    output = nd.concat(*[tid, tscore, left, top, right, bottom], dim=-1)
    out = nd.contrib.box_nms(output.reshape((0, -1, 6)))
    out = out.asnumpy()
    box = out[0][0][2:6] * np.array([image.shape[1], image.shape[2]] * 2)
    rect = box_to_rect(nd.array(box), 'green', 2)
    image = image.transpose((1,2,0))
    i0 = (image * rgb_std + rgb_mean).asnumpy()
    i0 = i0.clip(0, 255) / 255.
    plt.imshow(i0)
    plt.gca().add_patch(rect)
    plt.show()
    #plt.savefig('check_tbox.jpg')
    return box
def evaluate(net, n_epoch, n_iter, test_img, test_label):    
    test_out = predict(net, test_img)
    test_pred = test_out[:,0,-4:]
    test_lb = test_label.squeeze()[:,1:]	
    loss = nd.mean(l1_loss(test_pred, test_lb))
    print('Epoch: {0}    Iteration: {1}    Evaluation loss: {2}'.format(n_epoch, n_iter, loss))
    #plot_prediction(test_img, test_pred, n_epoch, n_iter)
    return loss



