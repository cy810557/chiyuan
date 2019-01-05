# coding: utf-8

import os
import cv2
import pdb
import argparse
import numpy as np
from datetime import datetime
from matplotlib.pyplot import *
from os.path import join

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, classes, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS):
    label = str(classes[class_id])+': {:.3f}'.format(confidence)
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


# In[4]:


def setup(class_txt, weights, config_file):
    classes = None
    #pdb.set_trace() 
    with open(class_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(weights, config_file)
    return net, classes


# In[5]:


def inference_single_image(net, image, classes, scale=0.00392, conf_threshold=0.5, nms_threshold = 0.4, COLORS=None):
    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, scale, (512, 512), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []    

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices)>0:
        #print('found object!')
        filename = '-'.join([ classes[i] for i in class_ids ])+'_'+datetime.now().strftime("%H-%M-%S.%f")+'.jpg' 
        filename_pred = 'pred_'+filename
        f = open('DataFactory/'+filename[:-4]+'.txt', 'w')
        cv2.imwrite('DataFactory/'+filename, image)
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, classes, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), COLORS)
            f.writelines('{0} {1} {2} {3} {4}\n'.format(class_ids[i], x/Width, y/Height, w/Width, h/Height))
        f.close()
        cv2.imwrite('DataFactory/'+filename_pred, image)


        
if __name__ == '__main__':
    if len(sys.argv)<3:
        print('Usage: python yolo_label_creator.py {1 | 0 (tiny or full)} {video name}')
        exit(0)
    
    model = int(sys.argv[1])
    video_name = sys.argv[2]
    verbose = False
    
    if model==1:
        weights = 'yolo_files/weights/tiny_yolo_24000_version_2.weights'
        cfg = 'yolo_files/cfg/tiny_yolo_version2.cfg'
    elif model==0: 
        weights = 'yolo_files/weights/yolov3_version_1.weights'
        cfg = 'yolo_files/cfg/yolov3.cfg'
    elif model==2:
        weights = 'yolo_files/weights/tiny_yolo_anchor_clustered_10000.weights'
        #cfg = 'yolo_files/cfg/tiny_yolo_anchor_clustered.cfg'
        cfg = 'yolo_files/cfg/tiny_yolo_version2.cfg'

    class_txt = 'yolo_files/extrem.names'
    
    net, classes = setup(class_txt, weights, cfg)
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    st = time.time()
    print(time.time()-st)
    video = cv2.VideoCapture(sys.argv[2])

    while True:
        # Read a new frame
        time.sleep(0.000001)
        timer = cv2.getTickCount()
        success, frame = video.read()
        if not success:
            # Frame not successfully read from video capture
            break
        
        inference_single_image(net, frame, classes, conf_threshold=0.85, COLORS=COLORS)
        
        if verbose:
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
            cv2.imshow("object detection", frame)
        
        k = cv2.waitKey(1) & 0xff
        if k == 27: break # ESC pressed       
    cv2.destroyAllWindows()
    video.release()


