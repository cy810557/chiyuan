# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import glob
from mxnet import nd
from skimage import io
import pickle
import mxnet as mx
import os
IMAGE_MEAN = nd.array([169.681,140.624,127.950]).reshape(1,1,3)

# def load_test_img(TEST_DIR):
#
#     img_list = glob.glob(TEST_DIR+'*.jpg')
#     num_instances=len(img_list)
#     for name in img_list:
#
def my_l2_loss(X,Y):
    num_instances=X.shape[0]
    return nd.sum(nd.square(X-Y))/2/num_instances


def model_initilze(params_dir,model,ctx):

    with open(params_dir+'params.pkl', 'rb') as file:
        p1 = pickle.load(file)
    # model = CPM(stages=6, joints=21)
    # model.hybridize()
    net_params = model.collect_params()

    # print len(net_params)
    print(len(p1))
    # i = 0
    for i, net_p in enumerate(net_params):
        # print w_p1[i]
        weight = mx.nd.array(p1[i].values()[0])
        if (i % 2) == 0:
            weight = mx.nd.transpose(data=weight, axes=(3, 2, 0, 1))
        net_params[net_p]._load_init(weight, ctx=ctx)

# def plot_hand_pts(img,hand_pts_preds):
#     '''
#     输入均为NDArray
#     '''
#     TYPE=mx.ndarray.ndarray.NDArray
#     preds=hand_pts_preds.asnumpy()
#     assert type(img) == TYPE or type(hand_pts_preds)==TYPE
#     hand_pts_cord = [np.where(preds[i]==np.max(preds[i])) for i in range(21)]
#     y_cord = [hand_pts_cord[i][0][0]*8 for i in range(21)]
#     x_cord = [hand_pts_cord[i][1][0]*8 for i in range(21)]
#     plot_feature_map(img)
#     plt.plot(x_cord,y_cord,'r.')

def get_hand_pts_per_instance(output_per_instance):
    '''
    输入为NDArray,shape:(None,21,46,46)
    输出：list，长度为None，list的每个元素为21个坐标点（X_i,Y_i）
    '''
    TYPE = mx.ndarray.ndarray.NDArray
    preds = output_per_instance.asnumpy()
    assert output_per_instance.shape == (21, 46, 46)
    locate_func = lambda x: np.array(np.where(x == np.max(x)))[:, 0]
    hand_pts_cord = map(locate_func, preds)
    hand_pts_cord = np.squeeze(np.array(hand_pts_cord)) * 8
    y_cord = hand_pts_cord[:, 0]
    x_cord = hand_pts_cord[:, 1]
    return x_cord, y_cord

def get_hand_pts(outputs):
    return map(get_hand_pts_per_instance, outputs)

def plot_hand_pts(img,outputs):
    '''
    输入均为NDArray
    '''
    edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],
             [9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    TYPE=mx.ndarray.ndarray.NDArray
    x_cord, y_cord =get_hand_pts_per_instance(outputs)
    plot_feature_map(img)
    plt.plot(x_cord,y_cord,'r.')
    for i,xy in enumerate(zip(x_cord, y_cord)):
        plt.annotate("%s" % i, xy=xy, xytext=(-8, -6), textcoords='offset points',fontsize=10)

def plot_feature_map(feature_NDArry):
    if len(feature_NDArry.shape)>2:
        feature_map=feature_NDArry.asnumpy()
        feature_map=np.rollaxis(feature_map,0,3)
        #pdb.set_trace()
        if np.mean(feature_map)<1:
            plt.imshow(np.uint8(feature_map*255))
        else:
            plt.imshow(np.uint8(feature_map))
        plt.axis('off')
        # plt.show()
    else:
        plt.imshow(np.uint8(feature_NDArry.asnumpy()*255))
        # plt.show()
        plt.axis('off')
def viz_feature_map(output,label=None,stage=6):
    if label is not None:
        #plt.clf()
        for i in range(4):
            plt.subplot(2,4,i+1)
            plot_feature_map(output[0][(stage-1)*22 + 5*(i+1)])
            plt.subplot(2,4,i+5)
            plot_feature_map(label[0][-22 +5*(i+1)])
        plt.show()
    else:
        plt.clf()
        for i in range(21):
            plt.subplot(3,7,i+1)
            plot_feature_map(output[i])
        #plt.show()
def save_stage_output(gpu_imgs,outputs,gpu_labels,iter,epoch, save_dir='viz_op_stages/'):

    # plot_hand_pts(gpu_imgs[0][0], outputs[0][0][110:131])
    # plt.savefig('viz_op_stages/epoch_{0}_iter_{1}_result.jpg'.format(epoch, iter))
    viz_feature_map(gpu_labels[0][0][22:44])
    plt.savefig(save_dir+'epoch_{0}_iter_{1}_label.jpg'.format(epoch,iter))
    viz_feature_map(outputs[0][0][0:22])
    plt.savefig(save_dir+'epoch_{0}_iter_{1}_op_s1.jpg'.format(epoch,iter))
    viz_feature_map(outputs[0][0][22:44])
    plt.savefig(save_dir+'epoch_{0}_iter_{1}_op_s2.jpg'.format(epoch,iter))
    viz_feature_map(outputs[0][0][44:66])
    plt.savefig(save_dir+'epoch_{0}_iter_{1}_op_s3.jpg'.format(epoch,iter))
    viz_feature_map(outputs[0][0][66:88])
    plt.savefig(save_dir+'epoch_{0}_iter_{1}_op_s4.jpg'.format(epoch,iter))
    viz_feature_map(outputs[0][0][88:110])
    plt.savefig(save_dir+'epoch_{0}_iter_{1}_op_s5.jpg'.format(epoch,iter))
    viz_feature_map(outputs[0][0][110:132])
    plt.savefig(save_dir+'epoch_{0}_iter_{1}_op_s6.jpg'.format(epoch,iter))

    print('Feature map visualizing and saving : Done!')
def pickle_NDArray(data,filename):
    assert filename.endswith('.pkl')
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


def l2_loss_with_zero_penaty(X, Y, delta):
    '''
    求出X,Y(shape均为(None,channel,height,weight))之间的l2-loss，并惩罚所有全零通道
    delta为penalty的影响系数。越小则越不考虑全零惩罚

    '''
    sigma = 1e-3  # 避免log(0)
    # zero_penalty_per_channel = lambda x:  - nd.log(nd.sum(x)+sigma)
    # zero_penalty_per_channel = lambda x: 1/(100*(nd.sum(x)+sigma))
    zero_penalty_per_channel = lambda x: 1 / (100 * nd.sum(x) + sigma)

    num_instances = X.shape[0]
    l2_loss = mx.nd.sum(mx.nd.square(X - Y)) / (2 * num_instances)
    channel_list = [X[i][j] for i in range(X.shape[0]) for j in range(X.shape[1])]  # loop all channels

    zero_penalties = sum([zero_penalty_per_channel(x) for x in channel_list])/num_instances
    total_loss = l2_loss + delta * zero_penalties

    return total_loss


def l2_loss_with_zero_penaty_v1(X, Y, delta):
    '''
    求出X,Y(shape均为(None,channel,height,weight))之间的l2-loss，并惩罚所有全零通道
    delta为penalty的影响系数。越小则越不考虑全零惩罚

    '''
    X, Y = X.asnumpy(), Y.asnumpy()
    sigma = 1e-3  # 避免log(0)
    zero_penalty_per_channel = lambda x: 1 / (100 * x + sigma)
    num_instances = X.shape[0]
    num_channels = X.shape[1]

    l2_loss = np.sum(np.square(X - Y)) / (2 * num_instances)

    zero_penaltiy_channels = np.vectorize(zero_penalty_per_channel, otypes=[np.float])  # 将对单个元素求倒数的函数apply到输入array各个元素
    # pdb.set_trace()
    zero_penalty = np.sum(
        zero_penaltiy_channels(np.sum(np.reshape(X, (num_channels * num_instances, -1)), axis=-1))) / num_instances
    # 问题：这里的zero_penalty最后是应该除以num_instance(平均每个样本的132个通道的penalty)，还是除以m*c（平均每个通道的penalty）？？？

    total_loss = l2_loss + delta * zero_penalty
    return nd.array(total_loss.ravel())


def l2_loss_with_zero_penaty_v2(X, Y, delta):
    '''
    求出X,Y(shape均为(None,channel,height,weight))之间的l2-loss，并惩罚所有全零通道
    delta为penalty的影响系数。越小则越不考虑全零惩罚

    '''
    # X,Y = X.asnumpy(),Y.asnumpy()
    sigma = 1e-3  # 避免log(0)
    zero_penalty_per_channel = lambda x: 1 / (100 * x + sigma)
    num_instances = X.shape[0]
    num_channels = X.shape[1]

    l2_loss = nd.sum(nd.square(X - Y)) / (2 * num_instances)

    # zero_penaltiy_channels = np.vectorize(zero_penalty_per_channel,otypes=[np.float]) #将对单个元素求倒数的函数apply到输入array各个元素
    # pdb.set_trace()
    sum_channels = nd.sum(nd.reshape(X, (num_channels * num_instances, -1)), axis=-1)
    zero_penalty = sum(map(zero_penalty_per_channel, sum_channels)) / num_instances
    # 问题：这里的zero_penalty最后是应该除以num_instance(平均每个样本的132个通道的penalty)，还是除以m*c（平均每个通道的penalty）？？？
    # pdb.set_trace()
    total_loss = l2_loss + delta * zero_penalty
    # pdb.set_trace()
    return total_loss

def create_mask(BATCH_SIZE,CTX):
    mask = nd.ones((BATCH_SIZE / 2, 132, 46, 46),ctx=CTX)
    mask[:, 21, :, :] = 0
    mask[:, 43, :, :] = 0
    mask[:, 65, :, :] = 0
    mask[:, 87, :, :] = 0
    mask[:, 109, :, :] = 0
    mask[:, 131, :, :] = 0
    return mask

def is_all_zero(data):
    num_channels=data.shape[0]
    zeros_count=0
    zero_list=[]
    for n in range(num_channels):
        if nd.sum(data[n])==0:
            zero_list.append('All zeros')
            zeros_count+=1
        else:
            zero_list.append('Not All zeros')
    print('{} channels are all-zero channel'.format(zeros_count))
    return zeros_count,zero_list

def single_channel_l2_loss(X,Y):
    return mx.nd.sum(mx.nd.square(X-Y))/2

def check_loss_components(pkl_name,num_stage):
    with open(pkl_name, 'rb') as f:
        op_lb_dict = pickle.load(f)
    outputs = op_lb_dict['outputs']
    gpu_labels = op_lb_dict['gpu_labels']
    op = outputs[0][15]
    op1 = op[:22]
    op2 = op[22:44]
    op3 = op[44:66]
    op4 = op[66:88]
    op5 = op[88:110]
    op6 = op[110:132]
    op_list=[op1,op2,op3,op4,op5,op6]
    lb = gpu_labels[0][0][:22]
    num_zeros,zero_list = is_all_zero(op_list[num_stage-1])
    for i in range(22):
        print(single_channel_l2_loss(op_list[num_stage-1][i], lb[i]), zero_list[i])

