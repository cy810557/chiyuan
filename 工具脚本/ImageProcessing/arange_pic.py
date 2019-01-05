#!/usr/bin/env python
# coding=utf-8
'''将用户输入的多张图像(4,9,16等)显示在一张图中，显示方式类似论文中多个图像紧密排列在一起。用户可以指定显示的行列以及padding间隔'''
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pdb
import argparse
from imutils import paths

args = argparse.ArgumentParser()
args.add_argument("-i", "--imgPaths")
args.add_argument("-r", "--rows", type=int,
                 help="rows of pics")
args.add_argument("-p", "--pad", type=int, default=0,
                 help="number of pixels to pad the boarder")
args = args.parse_args()

def arange_pics(imgPaths, rows, pad=0):
    imgs_array = []
    imgPaths = sorted(list(paths.list_images(imgPaths)))
    for imgPath in imgPaths:
        img = cv2.imread(imgPath)[...,::-1]
        img = cv2.resize(img, (64, 64))
        img = np.pad(img, ((pad, pad),(pad, pad),(0, 0)), mode="constant", constant_values=255)
        imgs_array.append(img)
    imgs_array = np.array(imgs_array)
    assert(imgs_array.shape[0]%rows==0), "number of rows cannot be divided by number of pictures."
    #pdb.set_trace()
    imgs_array = imgs_array.reshape((-1, rows) + imgs_array.shape[1:])
    imgs_array = np.hstack(np.hstack(imgs_array))
    return imgs_array

if __name__ =="__main__":
    # 最后图像四周也会有pad，该问题待解决
    tiled = arange_pics(args.imgPaths, int(args.rows), int(args.pad))
    plt.imshow(tiled);plt.axis("off");plt.savefig("tiled.png")


