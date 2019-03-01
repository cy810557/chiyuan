# -*- coding: UTF-8 -*-

import os
import sys
from xml.dom import minidom  #处理xml数据
from os.path import join
import argparse

args = argparse.ArgumentParser()
args.add_argument("-d", "--dir",
                 help="path to target dataset.")
args.add_argument("-l", "--lst",
                 help="name of .lst file to be written.")
args.add_argument("-c", "--classFile",
                 help="path to txt file given classNames.")
args = args.parse_args()

#首先定义一个读取xml文件的函数：
def xmlDecode(path, classNames):
    annotation = minidom.parse(path)

    size = annotation.getElementsByTagName('size')[0]
    width = size.getElementsByTagName('width')[0].firstChild.data
    height = size.getElementsByTagName('height')[0].firstChild.data

    obj = annotation.getElementsByTagName('object')[0]
    cla = obj.getElementsByTagName('name')[0].firstChild.data  #类别
    bndbox = obj.getElementsByTagName('bndbox')[0]              #坐标
    x1 = bndbox.getElementsByTagName('xmin')[0].firstChild.data
    x2 = bndbox.getElementsByTagName('xmax')[0].firstChild.data
    y1 = bndbox.getElementsByTagName('ymin')[0].firstChild.data
    y2 = bndbox.getElementsByTagName('ymax')[0].firstChild.data
    

    width = int(width)
    height = int(height)
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    result = [classNames.index(cla), (width,height), (x1,y1), (x2,y2)]
    return result

def getClassNames(txtFile):
    with open(txtFile, "r") as f:
        classNames = f.readlines()[0].split(" ")
    return list(filter(None, classNames))

if __name__ == "__main__":
    #定义保存数据和标签文件夹路径
    path = args.dir
    lst_file_name = args.lst
    #获取xml文件中出现的所有类别名称
    classNames = getClassNames(args.classFile)
    #假设图片名和对应的标签名称一致，这里直接替换xml为jpg
    #format:0  4  5  640(width)  480(height)  1(class)  0.1  0.2  0.8  0.9(xmin, ymin, xmax, ymax)  2  0.5  0.3  0.6  0.8  data/xxx.jpg
    names = os.listdir(path)
    lst = []
    i=0
    f = open(join(path,lst_file_name),'w')
    for name in names:
        if name.endswith('.xml'):
            result = xmlDecode(join(path, name), classNames)
            img_name = join(path, name.replace('xml','jpg'))
            lst_tmp =str(i)+'\t4'+'\t5'+'\t'+str(result[1][0])+'\t'+str(result[1][1])+'\t' \
            +str(result[0])+'\t' \
            +str(result[2][0]/result[1][0])+'\t'+str(result[2][1]/result[1][1])+'\t' \
            +str(result[3][0]/result[1][0])+'\t'+str(result[3][1]/result[1][1])+'\t' \
            +img_name+'\n' 
            f.write(lst_tmp)
            i+=1
    f.close()
