# -*- coding: utf-8 -*-

import cv2
import os
import pdb
import sys
import pyinter
import matplotlib.pyplot as plt

def rearrange(tp_lst):
    arrange_func = lambda x: float(x.split(':')[0]) * 60. + float(x.split(':')[1])
    new_lst = list(map(arrange_func, tp_lst))
    return new_lst

def pairwise(lst):
    '''lst代表任意支持迭代的对象'''
    it = iter(lst)
    while True:
        yield next(it), next(it)

def pairwise2(lst):
    '''另一种方法：利用zip实现多元素迭代'''
    for x,y in zip(lst[0::2], lst[1::2]):
        yield x,y

def get_time_intervals(time_points):
    inter_lst = []
    for start, end in pairwise(time_points):
        inter_lst.append(pyinter.closed(start, end))
    intervalSet = pyinter.IntervalSet(inter_lst)
    return intervalSet

def video2frame(video_path, pause_extract=1, save_prefix=None, intervalSet=None, frame_width=None, frame_height=None): #
    """
    将视频按固定间隔读取写入图片
    param video_name: 视频名称
    param save_prefix:　保存路径前缀
    param frame_width:　保存帧宽
    param frame_height:　保存帧高
    param pause_extract:　保存帧间隔
    return:　帧图片
    """
    video_name = os.path.basename(video_path)
    save_path = os.path.join(save_prefix , video_name[:-4])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    frame_index =  0   
    frame_count =  0   
    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")
        
    while(success):
        success, frame = cap.read()
        frame_index += 1
        #pdb.set_trace()
        if frame_index % pause_extract ==  0:
            if intervalSet is not None:
                time_point_now = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
                if time_point_now in intervalSet:
                    cv2.imwrite(save_path + "/%d.jpg" % frame_count, frame)
                    #cv2.putText(frame, str(time_point_now),(50,150),cv2.FONT_HERSHEY_COMPLEX, 6, (255,0,0),20)
                    #cv2.imshow("test_window",frame)   #---> DEBUG MODE
                    
            else:   #不使用intervalSet，直接按固定间隔对全视频抽帧
                cv2.imwrite(save_path + "/%d.jpg" % frame_count, frame)
                #cv2.imshow("test_window", frame)
        frame_count += 1        
        if cv2.waitKey(1) & 0xff ==27:
            break
    cv2.destroyAllWindows()
    cap.release()
    print('[+]Video {} processed done!'.format(video_name))

if __name__ == "__main__":
    if len(sys.argv)<2:
        print('Usage: python get_frame.py {video_path}, Optional: {extracting_interval}, {[time_points]}')
        exit()
    video_path = sys.argv[1]
    pause_extract = int(sys.argv[2])
    time_points = []
    for x in sys.argv[3:]:
        time_points.append(x)
    if len(time_points) > 0:
        assert len(time_points)%2==0,"输入的时刻数目应该为偶数!"
        time_points = rearrange(time_points)
        intervalSet = get_time_intervals(time_points)
    else:
        intervalSet = None
    video2frame(video_path, pause_extract=pause_extract, save_prefix='frames', intervalSet=intervalSet)
