import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import nd

def view_single(data, id):
    if data.shape[1] == 3:
        img = data[id].transpose((1,2,0)).asnumpy()
    else:
        img = data[id].asnumpy()
    plt.imshow(img)
    plt.show()
    return img


def imshow(x):
    if x.shape[0]==3:
        x = x.transpose((1,2,0))
    if isinstance(x, mx.ndarray.ndarray.NDArray):
        x = x.asnumpy()
    if x.dtype==np.float32:
        x = np.uint8(x)
    try:
        plt.imshow(x)
    except:
        print(x.shape)
    plt.show()


if __name__=="__main__":
    X_padded = nd.pad(X.transpose((0, 3, 1, 2)).astype(np.float32), mode='constant', pad_width=(0, 0, 0, 0, 4, 4, 4, 4))