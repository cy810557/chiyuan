from yolo_label_creator import *
import cv2
import sys
from os.path import join
if len(sys.argv)<2:
    print('Usage: python debug_inference_single_img.py 0(full yolo) | 1(tiny yolo) | 2(clusterd tiny)')
    exit(0)
model = int(sys.argv[1])
if model==1:
    weights = 'yolo_files/weights/tiny_yolo_24000_version_2.weights'
    cfg = 'yolo_files/cfg/tiny_yolo_version2.cfg'
elif model==0: 
    weights = 'yolo_files/weights/yolov3_version_1.weights'
    cfg = 'yolo_files/cfg/yolov3.cfg'
elif model==2:
    weights = 'yolo_files/weights/tiny_yolo_24000_version_2.weights'
    cfg = 'yolo_files/cfg/tiny_yolo_anchor_clustered.cfg'
class_txt = 'yolo_files/extrem.names'
#pdb.set_trace()
net, classes = setup(class_txt, weights, cfg)
frame = cv2.imread('baby_carriage_2_5_885.jpg')
assert frame is not None,"frame not read!"
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
inference_single_image(net, frame, classes, conf_threshold=0.2, COLORS=COLORS)
