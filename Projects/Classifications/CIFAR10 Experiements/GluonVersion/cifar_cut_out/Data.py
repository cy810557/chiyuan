import mxnet as mx
from mxnet import nd, image
import numpy as np
from mxnet.gluon import data as gdata

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        assert(img.shape[0]==3), "Input to before cutout should be C x H x W., given: {}".format(img.shape)
        h = img.shape[1]
        w = img.shape[2]

        mask = np.ones((h, w), np.uint8)  # np.float32

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0

        mask = nd.tile(nd.array(mask), (3 ,1 ,1)  )# .transpose((1,2,0))  #再次用到tail函数
        # 为什么这时候mask的类型是float32?
        return img.astype('float32') * mask

class Pad(object):
    def __init__(self, pad_width=(4 ,4 ,0)):
        self.pad_width =pad_width
    def __call__(self, img):
        pw = self.pad_width
        assert(img.shape[2]==3), "Input to before cutout should be H x W x C."
        img = img.asnumpy()
        img = np.pad(img, ((pw[0] ,pw[0]) ,(pw[1] ,pw[1]) ,(pw[2] ,pw[2])) ,mode='constant')
        return nd.array(img)

def load_dataset(Train_Root, Val_Root, Train_Val_Root=None, batch_size=128, trans_train=None, trans_test=None):
    train_ds = gdata.vision.ImageFolderDataset(Train_Root) #transform = transform  # flag默认为1， 加载RGB图像
    valid_ds = gdata.vision.ImageFolderDataset(Val_Root)
    # total_ds = gdata.vision.ImageFolderDataset(Train_Val_Root, flag=1)  # 调好参数之后在整个数据集上训练模型并提交测试结果

    train_iter = gdata.DataLoader(train_ds.transform_first(trans_train),
                                  batch_size, shuffle=True, last_batch='keep')
    val_iter = gdata.DataLoader(valid_ds.transform_first(trans_test),
                                batch_size, shuffle=True, last_batch='keep')
    # total_iter = gdata.DataLoader(total_ds.transform_first(transform_train),
    #                              batch_size, shuffle=True, last_batch='keep')
    return train_iter, val_iter

def transform_train(data, label):
    im = data.asnumpy()
    im = np.pad(im, ((4,4),(4,4),(0,0)), mode='constant')
    # 前半部分为np augment. 后半部分为nd augment，注意要求转换为float型
    im = nd.array(im, dtype="float32") / 255
    auglist = image.CreateAugmenter(
        data_shape=(3,32,32), resize=0, rand_mirror=True, rand_crop=True,
        mean=nd.array([0.4914, 0.4822, 0.4465]), std=nd.array([0.2023, 0.1994, 0.2010])
    )
    for aug in auglist:
        im = aug(im)
    im = im.transpose((2, 0, 1))
    return im, label.astype('float32')

def transform_test(data, label):
    im = data.astype("float32") / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                    std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, label.astype('float32')