# -*- coding: utf-8 -*-

import os
import cv2
import sys
'''
计算整个文件夹内图像的R,G,B三通道均值和标准差
'''
TARGET_DIR = sys.argv[1]
img_list=os.listdir(TARGET_DIR)
img_list = [i for i in img_list if i.endswith('.jpg')]
img_size=256
sum_r=0
sum_g=0
sum_b=0
std_r=0
std_g=0
std_b=0
count=0

for img_name in img_list:
    img_path=os.path.join(TARGET_DIR,img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(img_size,img_size))
    sum_r=sum_r+img[:,:,0].mean()
    sum_g=sum_g+img[:,:,1].mean()
    sum_b=sum_b+img[:,:,2].mean()
    std_r=std_r+img[:,:,0].std()
    std_g=std_g+img[:,:,1].std()
    std_b=std_b+img[:,:,2].std()
    
    count=count+1

sum_r=sum_r/count
sum_g=sum_g/count
sum_b=sum_b/count
std_r=std_r/count
std_g=std_g/count
std_b=std_b/count
img_mean=[sum_r,sum_g,sum_b]
img_std=[std_r,std_g,std_b]
print('R,G,B mean: ', img_mean)
print('R,G,B std: ', img_std)
