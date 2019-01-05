# coding: utf-8
import os
import pdb 
import time
import logging
import mxnet as mx
import numpy as np
from os.path import join
from mxnet.gluon import nn
import matplotlib.pyplot as plt
from mxnet import gluon, nd, autograd
from mxnet.gluon.nn import Conv2D, LeakyReLU, BatchNorm, Dropout, Activation, Conv2DTranspose
from datetime import datetime



# ### Step 0: Data Preprocessing  
# * 这里使用和上一节类似的`os.walk + mx.io.NDArrayIter`的套路

# In[2]:



def preprocess_single_img(img):
    assert isinstance(img, mx.ndarray.ndarray.NDArray), "input must be NDArray type."
    img = mx.image.imresize(img, 2 * img_wd, img_ht)
    img_in = img[:,:img_wd].transpose((2,0,1)).expand_dims(0)
    img_out = img[:,img_wd:].transpose((2,0,1)).expand_dims(0)
    assert img_in.shape==(1, 3, 256, 256), "image shape not correct."
    return img_in, img_out
    
def load_data(data_path, batch_size, reverse=False):
    img_in_list, img_out_list = [], []
    for path, _, files in os.walk(data_path):
        for file in files:
            if not file[-4:] in ['.jpg']:
                continue
            img_arr = mx.image.imread(join(path, file)).astype(np.float32)/127.5 - 1
            img_in, img_out = preprocess_single_img(img_arr)
            if not reverse:
                img_in_list.append(img_in)
                img_out_list.append(img_out)
            else:
                img_in_list.append(img_out)
                img_out_list.append(img_in)
    return mx.io.NDArrayIter(data = [nd.concatenate(img_in_list), nd.concatenate(img_out_list)], batch_size=batch_size)

def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
def show_samples(data_iter, num_samples=4):
    img_in_list, img_out_list = data_iter.next().data
    for i in range(num_samples):
        plt.subplot(2, num_samples, i+1)
        visualize(img_in_list[i])
        plt.subplot(2, num_samples, i+num_samples+1)
        visualize(img_out_list[i])
    plt.show()



# ### Step 2 Network Design  
# #### 2.1 Unet网络块定义：  
# 注意：
# ①同TensorFlow一样，卷积层默认: use_bias=True  
# ②Unet Block定义的基本思路：先定义好Encoder-Decoder结构，最后再hybrid_forward中将Encoder-Decoder输入特征级联到输出特征即可  
# ③**除了最内层的Block，其他所有层，输入Encoder的feature map的通道数都是输入Decoder的feature map通道数的一半**  
# ④BatchNorm层默认参数设置momentum为0.9，而G和D中都设置为0.1？

# In[6]:


class UnetSkipUnit(nn.HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False, use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()
        # 先定义一些基本的组件
        self.outermost = outermost
        en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1, in_channels=outer_channels, use_bias=use_bias)
        en_relu = LeakyReLU(alpha=0.2)
        en_bn = BatchNorm(momentum=0.1, in_channels=inner_channels)
        deconv_innermost = Conv2DTranspose(
            channels=outer_channels, kernel_size=4, strides=2, padding=1, in_channels=inner_channels, use_bias=use_bias)
        deconv_output = Conv2DTranspose(
            channels=outer_channels, kernel_size=4, strides=2, padding=1, in_channels=2*inner_channels, use_bias=True)
        deconv_common = de_conv_innermost = Conv2DTranspose(
            channels=outer_channels, kernel_size=4, strides=2, padding=1, in_channels=2*inner_channels, use_bias=use_bias)

        de_relu = Activation('relu')
        de_bn = BatchNorm(momentum=0.1, in_channels=outer_channels)
        # Unet网络块可以分为三种：最里面的，中间的，最外面的。
        if innermost:
            encoder = [en_relu, en_conv]
            decoder = [de_relu, deconv_innermost, de_bn]
            model = encoder + decoder
        elif outermost:
            encoder = [en_conv]
            decoder = [de_relu, deconv_output]
            model = encoder + [inner_block] + decoder
            model += [Activation('tanh')]
        else:
            encoder = [en_relu, en_conv, en_bn]
            decoder = [de_relu, deconv_common, de_bn]
            model = encoder + [inner_block] + decoder
        if use_dropout:
            model += [Dropout(0.5)]
        self.model = nn.HybridSequential()
        with self.model.name_scope():
            for block in model:
                self.model.add(block)
    def hybrid_forward(self, F, x):
        # 除了outermost之外的block都要加skip connection
        if self.outermost:
            return self.model(x)
        else:
            #pdb.set_trace()
            return F.concat(self.model(x), x, dim=1)


class UnetGenerator(nn.HybridBlock):
    def __init__(self, input_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()
        unet= UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
        unet = UnetSkipUnit(ngf, input_channels, unet, outermost=True)
        self.model = unet
    def hybrid_forward(self, F, x):
        return self.model(x)




class Discriminator(nn.HybridBlock):
    def __init__(self, in_channels, n_layers=3, ndf=64, use_sigmoid=False, use_bias=False):
        super(Discriminator, self).__init__()
        # 用下面一段代码来配置标准的2x 下采样卷积
        kernel_size=4
        padding = int(np.ceil((kernel_size-1)/2))
        self.model = nn.HybridSequential()
        # 先用一个卷积将输入转为第一层feature map
        self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2, padding=padding, use_bias=use_bias, in_channels=in_channels))
        self.model.add(LeakyReLU(alpha=0.2))
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.model.add(
                Conv2D(channels=ndf*nf_mult, kernel_size=kernel_size, strides=2, padding=padding, use_bias=use_bias, in_channels=ndf*nf_mult_prev),
                BatchNorm(momentum=0.1, in_channels=ndf*nf_mult),
                LeakyReLU(alpha=0.2))
        
        # 若layers较少，channel未达到512， 可以继续升一点维度
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.model.add(
            Conv2D(channels=ndf*nf_mult, kernel_size=kernel_size, strides=1, padding=padding, use_bias=use_bias, in_channels=ndf*nf_mult_prev),
            BatchNorm(momentum=0.1, in_channels=ndf*nf_mult),
            LeakyReLU(alpha=0.2))
        # 输出： output channel为什么设为1？
        self.model.add(Conv2D(channels=1, kernel_size=kernel_size, strides=1, padding=padding, use_bias=True, in_channels=ndf*nf_mult))
        if use_sigmoid:
            self.model.add(Activation('sigmoid'))
    def hybrid_forward(self, F, x):
        return self.model(x)


# ### 2.4 Construct Network  
# 注意：  
# ①这里的loss使用binary_cross_entropy + L1 loss 作为最终的loss。L1 loss用来capture 图像中的low frequencies  
# ②使用自定义的初始化方式:  (这里说的初始化均为实值初始化，而不是仅仅定义初始化方式)
# - 卷积层：
#    - $weight$: 标准差为0.02的高斯随机初始化
#    - $bias$: 全零初始化
# - BN层：
#    - 除了$gamma$之外，所有的bn参数（$beta, running__mean, running__var$）初始化为0； $gamma$: **均值为1**，标准差0.02的高斯随机初始化  
# 
# ③这里设置的Trainer中的beta1参数是bn中的吗？bn中的beta不应该是参数而不是超参数吗？ 答：是Adam中的第一动量 $β_1$

# In[9]:



def init_param(param):
    if param.name.find('conv') != -1: # conv层的参数，包括w和b
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('batchnorm') != -1: #bn层的参数
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        if param.name.find('gamma')!=-1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))
def network_init(net):
    for param in net.collect_params().values():
        init_param(param)
# 正式定义网络架构
def set_networks(num_downs=8, n_layers=3, ckpt=None):
    netG = UnetGenerator(input_channels=3, num_downs=8)
    netD = Discriminator(in_channels=6, n_layers=3)
    if ckpt is not None:
        print('[+]Loading Checkpoints {} ...'.format(ckpt))
        netG.load_parameters(ckpt+'G.params', ctx=ctx)
        netD.load_parameters(ckpt+'D.params', ctx=ctx)
        print('[+]Checkpoint loaded successfully!')
    else:
        network_init(netG)
        network_init(netD)
    
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate':lr, 'beta1':beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate':lr, 'beta1':beta1})
    
    return netG, netD, trainerG, trainerD

###################### Set loss function #######################



# ## Step 3: Training Loop

# ### 3.1 为判别模型专门定义一个ImagePool，使得判别模型不仅仅比较当前的真实输入和虚假输出的损失，还要考虑历史损失  
# * 理解：  
# 
# 首先在pool满之前，读入的每张图像都会被存储在pool的images成员变量中。同时也会返回一份给ret，用于传递到函数外面。  
# pool中只能存50张images，很快就会被占满。当pool满了以后，再query一个样本时，pool可能以百分之五十的几率选择如下两种操作中的一个:  
#   
# ①使用读入的image替换掉images列表中的随机一张，替换得到的images中的old image被分给ret，随后返回。  
# ②新的image被加入到ret中，pool中的images列表不更新  
# * 问题：  
# 
# 
# ①ImagePool的作用是什么？  
# ②pool会对每张图像进行qurey操作，最起码有一些nd运算。这会对训练的迭代速度产生多大的影响？

# In[12]:


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        ret_imgs = []
        for i in range(images.shape[0]):
            image = nd.expand_dims(images[i], axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                ret_imgs.append(image)
            else:
                p = nd.random_uniform(0, 1, shape=(1,)).asscalar()
                if p > 0.5:
                    random_id = nd.random_uniform(0, self.pool_size - 1, shape=(1,)).astype(np.uint8).asscalar()
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    ret_imgs.append(tmp)
                else:
                    ret_imgs.append(image)
        ret_imgs = nd.concat(*ret_imgs, dim=0)
        return ret_imgs

def facc(label, pred):
    return ((pred.ravel()>0.5) == (label.ravel())).mean()
def train(lamda=100, lr_decay=0.2, period=50, ckpt='.', viz=False):
    image_pool = ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)
    
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    #fig = plt.figure()
    for epoch in range(num_epochs):
        epoch_tic = time.time()
        btic = time.time()
        train_data.reset()
        for iter, batch in enumerate(train_data):
            real_in, real_out = batch.data[0].as_in_context(ctx), batch.data[1].as_in_context(ctx)
            fake_out = netG(real_in)
            fake_concat = image_pool.query(nd.Concat(real_in, fake_out, dim=1))
            with autograd.record():
                # Train with fake images
                output = netD(fake_concat) #?????????????????? 这里把x和fake一同送入D，是Conditional GAN的体现？如何理解这里的条件概率？
                fake_label = nd.zeros(output.shape, ctx=ctx)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label,],[output,])  ## metric应该何时update？？？
                
                # Train with real images
                real_concat = image_pool.query(nd.Concat(real_in, real_out, dim=1))
                output = netD(real_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD = (errD_fake + errD_real) * 0.5 ## 如论文所述，D loss乘以0.5以降低相对G的更新速率
                errD.backward()
                metric.update([real_label,],[output,])
            trainerD.step(batch_size)
            
            with autograd.record():
                fake_out = netG(real_in)    # 这里的G为什么没有体现出Conditional GAN？？  ####### 重要 #######
                #fake_concat = image_pool.query(nd.Concat(real_in, fake_out, dim=1))
                # 注意：image_pool只用于记录判别器
                fake_concat = nd.Concat(real_in, fake_out)  # Conditional GAN的先验：real_in，即 x
                output = netD(fake_concat)
                errG = GAN_loss(output, real_label) + lamda * L1_loss(real_out, fake_out)
                errG.backward()
            trainerG.step(batch_size)
            
            if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('Epoch {}, lr {:.6f}, D loss: {:3f}, G loss：{:3f}, binary training acc: {:2f}, at iter {}, Speed: {} samples/s'.format(
                epoch, trainerD.learning_rate,  errD.mean().asscalar(), errG.mean().asscalar(), acc, iter, 0.1*batch_size/ (time.time()-btic)))
            btic = time.time()
        if epoch % period == 0:
            trainerD.set_learning_rate(trainerD.learning_rate * lr_decay)
            trainerG.set_learning_rate(trainerG.learning_rate * lr_decay)
        if epoch % 100 == 0:
            print('[+]saving checkpoints to {}'.format(ckpt))
            netG.save_parameters(join(ckpt, 'pixel_netG_epoch_{}.params'.format(epoch)))
            netD.save_parameters(join(ckpt, 'pixel_netD_epoch_{}.params'.format(epoch)))
        name, epoch_acc = metric.get()
        metric.reset()
        logging.info('\n[+]binary training accuracy at epoch %d %s=%f' % (epoch, name, epoch_acc))
        logging.info('[+]time: {:3f}'.format(time.time() - epoch_tic))


if __name__=='__main__':
#### 超参数列表  
    ctx = mx.gpu(0)
    lr = 0.001
    batch_size = 10
    beta1 = 0.5  ## beta_1（第一动量）默认设置是0.9，为什么这里差别也这么大？？？
    pool_size = 50
    num_epochs = 500
    Dataset_Path= 'CMP_Dataset/facades/'
    img_wd, img_ht = 256, 256

    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L1Loss()

    netG, netD, trainerG, trainerD = set_networks(n_layers=2, ckpt='pixel_net')
    train_data = load_data(join(Dataset_Path,'train'), batch_size, reverse=True)
    val_data = load_data(join(Dataset_Path,'val'), batch_size, reverse=True)

    train(lamda=100, lr_decay=0.8, period=50, ckpt='pix2pix/models')
    print('[+]Training complete. Saving parameters...')
    netG.save_parameters('pixel_netG.params')
    netD.save_parameters('pixel_netD.params')
