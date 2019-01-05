!#/usr/bin/env
import os
import glob
Data_path = '/home/xie/Chiyuan/ChiyuanData/data2'
Original_path = '/home/xie/AAR_Throax/Original'
with open(Data_path+'/data_info.INFO') as f:
    info = f.read().split('\n')
    info.remove('')
file_glob = os.path.join(Data_path,'*.mat')
file_list = glob.glob(file_glob)
count = 0
for file in file_list:
    this_info = info[count].split(' ')
    slice_width = int(this_info[1])
    slice_num = int(this_info[2])
    x = float(this_info[3])
    z = float(this_info[4])
    name = file.split('/')[-1].split('.')[0]
    os.system('mkdir '+Original_path+'/'+name)
    command = 'importMath '+file+' matlab '+Original_path+'/{0}/{0}.IM0 {1} {1} {2} {3} {3} {4}'.format(name,slice_width, slice_num, x,z)
    os.system(command)
    count +=1

