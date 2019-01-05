if [[ $# == 0 ]]
then
    ./darknet detector train cfg/yolo_dms.data cfg/yolo_dms.cfg backup/darknet53.conv.74 -gpus 2 2>&1 | tee dms_yolo.log
elif [ "${1}" = "continue" ]
then
    echo "Continue training..."
    ./darknet detector train cfg/yolo_dms.data cfg/yolo_dms.cfg ../YOLO_DMS/backup/yolo_dms.backup -gpus 1 2>&1 | tee -a dms_yolo.log  
else  
    echo "[+]Usage: bash train_yolo.sh continue[+]"
fi
