
# coding: utf-8

# 该脚本尝试使用gluon搭建一个object localizator:  
# 读取图像，使用浅层神经网络回归至五维向量，以获取单个目标的置信度以及位置信息

# In[111]:


import os
import sys
import pdb
import mxnet as mx
import numpy as np
import gluonbook as gbk
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
from mxnet import gluon, nd, image, autograd
import matplotlib.pyplot as plt
from mxnet import init
#get_ipython().run_line_magic('matplotlib', 'inline')


# 主要使用mxnet中专门用于目标检测任务的迭代器：ImageDetIter  
# ** class mxnet.image.ImageDetIter ** (batch_size, data_shape, path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None, shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, data_name='data', label_name='label', **kwargs)[source]

# In[37]:


def data_iterator(batch_size, data_shape, data_root):
    #os.chdir(data_root)
    train_iter = image.ImageDetIter(
        batch_size = batch_size,
        data_shape = (3, data_shape, data_shape),
        path_imgrec = data_root+'/train.rec',
        path_imgidx = data_root+'/train.idx',
        shuffle = True,
        mean = True ,
        std = True,
        rand_crop = 1,
        min_object_covered = 0.9,
        max_attempts =  200)
    val_iter = image.ImageDetIter(
        batch_size = batch_size,
        data_shape = (3, data_shape, data_shape),
        path_imgrec = data_root+'/val.rec',
        path_imgidx = data_root+'/val.idx',
        shuffle = False,
        mean = True, 
        std = True)
    return train_iter, val_iter


# In[38]:


BATCH_SIZE = 32
DATA_SHAPE = 256
DATA_ROOT = sys.argv[1]
train_iter, val_iter = data_iterator(batch_size=32, data_shape=256, data_root = DATA_ROOT)


# In[40]:


batch = train_iter.next()
print(batch)


# In[61]:


def box_to_rect(box, color, linewidth=3):
    box = box.asnumpy()
    #pdb.set_trace()
    return plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1], 
                         fill=False, edgecolor=color, linewidth=linewidth)


# In[62]:


RGB_MEAN = nd.array([123, 117, 104])
RGB_STD = nd.array([58.395, 57.12, 57.375])

# _, figs = plt.subplots(3, 3, figsize=(9,9))
# for i in range(3):
    # for j in range(3):
        # img, label = batch.data[0][3*i+j], batch.label[0][3*i+j]
        # img = img.transpose((1,2,0)) * RGB_MEAN + RGB_STD
        # img = img.clip(0,255).asnumpy()/255
        # fig = figs[i][j]
        # fig.imshow(img)
        # rect = box_to_rect(label[0][1:5]*DATA_SHAPE , 'red', 2)
        # fig.add_patch(rect)
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
# plt.show()


# Step 2: 定义网络模型

# In[106]:


pretrained = vision.resnet18_v1(pretrained=True).features
net = nn.HybridSequential()
for i in range(8):
    net.add(pretrained[i])
net.add(nn.GlobalAvgPool2D())
Dense_layer = nn.Dense(units=4)
Dense_layer.initialize()
net.add(Dense_layer)

#net.load_parameters('models/model_5.params')
net


# In[112]:


#check_result:
test_input =  batch.data[0]
# test_input = test_input.expand_dims(axis = 0)
test_feature = net(test_input)
test_feature.shape


# 定义loss函数：

# In[113]:


l1_loss = gluon.loss.L1Loss()


# 定义训练函数

# In[110]:


ctx = mx.gpu(2)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':0.01, 'wd':5e-4})

def transform_center(box):
    box = nd.squeeze(box)
    x1, y1, x2, y2 = box.split(num_outputs=4, axis=-1)
    x,y = (x1+x2)/2, (y1+y2)/2
    w,h = x2-x1, y2-y1
    return nd.concat(*[x,y,w,h], dim=-1)
def feature_forward(feature):
    feature = nd.squeeze(feature)
    xywh_pred = nd.sigmoid(feature)
    return xywh_pred

NUM_EPOCHS = 40

for epoch in range(NUM_EPOCHS):
    train_iter.reset()
    for i, batch in enumerate(train_iter):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            feature = net(x)
            #with autograd.pause():
            xywh_pred = feature_forward(feature)

            box_lb = nd.slice_axis(y, begin=1, end=5, axis=-1)
            xywh_lb = transform_center(box_lb)
            loss = l1_loss(xywh_pred, xywh_lb)

        loss.backward()
        trainer.step(BATCH_SIZE)
        
        if i % 50 ==0:
            try:
                #pdb.set_trace()
                print(xywh_pred.asnumpy()[0], xywh_lb.asnumpy()[0])
                loss_record = nd.mean(loss).asscalar()
                print('Epoch: {0}, iter: {1}, loss: {2}'.format(
                    epoch, i, loss_record))
            except:
                #pdb.set_trace()
                print(xywh_pred.asnumpy()[0], xywh_lb.asnumpy()[0])
                loss_record = nd.mean(loss).asscalar()
                print('Epoch: {0}, iter: {1}, loss: {2}'.format(
                    epoch, i, loss_record))
    net.save_parameters('models/model_{}.params'.format(epoch))

