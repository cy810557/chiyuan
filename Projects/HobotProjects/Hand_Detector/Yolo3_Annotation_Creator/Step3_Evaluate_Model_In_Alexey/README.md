1. To inference a data batch, first create a image_list.txt which records the absolute path of each image.
2. Then excute SCRIPT: inferece_img_batches.sh to create the predictions of given dataset in a txt_list format.
3. Use the notebook to visualize images and corresponding labels created by yolo3.
4. Evaluate the performance of trained model: Calculate mAP, F1-Score, average IoU etc. :
./darknet detector map data/yolo_dms.data cfg/yolo_dms.cfg my_yolov3_8000.weights
