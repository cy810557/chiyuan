#!/usr/bin/env python
# coding=utf-8
from skimage.io import imread 
from imutils.paths import list_images
import os
from os.path import join
import argparse
import pdb

args = argparse.ArgumentParser()
args.add_argument("-img","--imgDir",
                 help="path to images.")
args.add_argument("-lbl","--lblDir",
                 help="path to labels.")
args.add_argument("-lst","--lstName",
                 help="name of .lst file.")
args = args.parse_args()

imgPaths = sorted(list(list_images(args.imgDir)))        
lblPaths = sorted(os.listdir(args.lblDir))                                                                                                                                                                                                    

f = open(args.lstName, "w")                                   
for i, (im, lb) in enumerate(zip(imgPaths, lblPaths)):           
    img = imread(im)                                     
    h, w = img.shape[:2]                                 
    with open(join(args.lblDir, lb), 'r') as f1:                             
        txt_lines = f1.readlines()                        

    line_info = "{}\t4\t5\t{}\t{}".format(i, h, w)
    for line in txt_lines:                               
        line = line.strip("\n").split(" ")  #string->list       
        line_info += ("\t{}"*5).format(*line) 
    f.write(line_info+ "\t" + im + "\n")                                  
f.close()                                                 
