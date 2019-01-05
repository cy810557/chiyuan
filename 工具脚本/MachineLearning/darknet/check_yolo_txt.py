# -*- coding: utf-8 -*-
# 检查darknet yolo数据集标注是否正确。脚本支持两种模式：单张检查和目录检查，后者将遍历并显示每个样本的图像及其标注信息
import cv2
import os
from os.path import join
import sys
import pdb
import matplotlib.pyplot as plt
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path of image or image_folder")
parser.add_argument("-f","--filter", default=None, help="keyword, item who contains it will be exclueded.")
parser.add_argument("-p","--pause",default=0.1, type=float, help="pause when plotting")
args = parser.parse_args()


def transform_box(box, width, height):
    w, h = box[2]*width, box[3]*height
    x = box[0]*width - w/2
    y = box[1]*height - h/2
    return [x, y, w, h]

def show_single_sample(img_file):

    with open(join(label_dir, img_file[:-4]+'.txt'),'r') as f:
        content = f.readlines()
    category = [float(c.split()[0]) for c in content]
    content=[c.split()[1:] for c in content]
    boxes = [[float(x) for x in c] for c in content]
    img = cv2.imread(join(image_dir, img_file))
    height, width = img.shape[:-1]
    for i,box in enumerate(boxes):
        box = transform_box(box, width, height)
        #print(box)
        #pdb.set_trace()
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]+ box[2]), int(box[1]+box[3])), (0,0,255), 3)
        cv2.putText(img, CLASS[int(category[i])], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255), 5)
    plt.imshow(img[...,::-1])
    plt.show()
def show_samples(img_files):
    plt.ion()
    for img_file in img_files:
        plt.clf()
        show_single_sample(img_file)
        plt.pause(args.pause) 

if __name__ == '__main__':
    path = args.path
    keyword = args.filter
    image_dir = 'images_trolley'
    label_dir = 'labels'
    CLASS = ['baby_carriage', 'trolley_case', 'cart']
    if path.endswith('.jpg'):
        show_single_sample(path)
    else: 
        img_files = os.listdir(path)
        if keyword is not None:
            img_files = list(filter(lambda x: keyword not in x, img_files))
        random.shuffle(img_files) 
        show_samples(img_files)



