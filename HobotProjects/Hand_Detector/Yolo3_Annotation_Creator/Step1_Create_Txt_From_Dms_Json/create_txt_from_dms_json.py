# coding:utf-8
import os
import sys
import pdb
from lxml import etree, objectify
from glob import glob
import dms_json
import dataset_hand

OUTPUT_DIR = '/mnt/hdfs-data-2/data/chengyuan.yang/DMS/Prepare_YOLO_Annotition/Outputs'
TRAIN_TXT_LIST = []

def convert_bbox(box):
    '''原box格式: xmin,ymin,xmax,ymax
       输出box格式：yolo格式：x_center/W, y_center/H, w/W, h/H
    '''
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x, w, y, h = x/1280, w/1280, y/720, h/720
    return [x,y,w,h]
def create_txt_one_line(line, img_folder_name, saving_dir):
    '''该函数处理一个json文件的一行（即一张图像），读取dms_json中该图像所有的hand_box信息，输出到一个与图像同名的txt中'''
    if line[0] == '#':
        return
    image_arrt = dms_json.DMSJsonParser()
    try:
        if image_arrt.ParseJsonRaw(line) == False:
            return
    except:
        return
    #pdb.set_trace()
    if image_arrt.hasDriver == False:
        return
    if len(image_arrt.driverLandmark) == 0:
        return
    if image_arrt.handNum == 0:
        return
    img_name = image_arrt.imgName 
    if img_name is None:
        return
    bboxes = image_arrt.handBoxes
    if len(bboxes)==0:
        return
    txt_file = open(saving_dir+'/'+img_name[:-4]+'.txt', 'w')
    for bbox in bboxes:
        bbox = convert_bbox(bbox)
        txt_file.write('0 '+' '.join([str(x) for x in bbox])+'\n')
    TRAIN_TXT_LIST.append(os.path.join(img_folder_name,img_name))
#     print("Successfully write txt")
def create_txt_one_json(json_file_name, data_name, img_dir):
    '''遍历一个json文件所有行'''
    with open(json_file_name,'r') as f:
        json_file = f.readlines()
    base_file_name = os.path.basename(json_file_name)
    json_fn_pfx = os.path.splitext(base_file_name)[0]
    img_folder_name = os.path.join(img_dir, json_fn_pfx)
    saving_dir = os.path.join(OUTPUT_DIR, data_name,json_fn_pfx)  # Project_Dir/Patch_name/5355/
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    for line in json_file:
        _=create_txt_one_line(line, img_folder_name, saving_dir)
    print('Successfully processed json file : {}'.format(json_fn_pfx))
def create_txt_one_patch(data_name):
    '''遍历一个data_patch(如patch_201805这种)的所有json文件'''
    json_dir, img_dir = dataset_hand.GetDatabase(data_name, data_type='train')
    json_file_list = glob(json_dir + '/*.json')
    for json_file_name in json_file_list:
        create_txt_one_json(json_file_name, data_name, img_dir)
    print('[+][+] Successfully processed data patch : {} [+][+]'.format(data_name))

    
patch_list = ['patch_201805','patch_201806','tianjin201806','plus_20180622','pts72_refresh','plus_20180709',
             'tianjin201806_plus_20180718','tianjin201806_plus_20180702','tianjin201806_plus_201808',
             'pts72_refresh_plus_20180702','pts72_refresh_plus_20180718','pts72_refresh_plus_201808']

# 这里只使用了patch_201805作为例子。如果要使用全部patch，使用整个patch_list即可
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python create_txt_from_dms_json.py {data_patch_name}')
        exit()
    if sys.argv[1] not in patch_list:
        print('Data name not valid! Please choose data name in \n {}'.format(patch_list))
        exit()
    create_txt_one_patch(data_name=sys.argv[1])    
    train_lst = open(OUTPUT_DIR+'/train_list.txt','w')
    for line in TRAIN_TXT_LIST:
        train_lst.write(line)
        train_lst.write('\n')
    train_lst.close()    
