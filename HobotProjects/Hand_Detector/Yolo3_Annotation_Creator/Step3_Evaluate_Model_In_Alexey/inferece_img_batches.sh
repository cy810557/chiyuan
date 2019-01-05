# usage: ./darknet detector test data/myv3_det.data  cfg/my_yolov3.cfg my_yolov3_8000.weights -dont_show -ext_output < My_output/2007_test.txt > My_output/result.txt
./darknet detector test data/myv3_det.data  cfg/my_yolov3.cfg ../project_yolo_v3/backup/my_yolov3_20000.weights -dont_show -ext_output < ${1} > ${2}

