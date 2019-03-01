#!/bin/bash
if [ ! $# -eq 3 ]; then
    echo "[INFO] Usage: bash prepare_rec_yolo.sh {YoloImageDir} {YoloLabelDir} {LstFileName}\n
    For example: bash prepare_rec_yolo.sh Dataset/images Dataset/labels hand_train.lst";
    exit 0;
fi

echo "[INFO] Creating .lst file..."
python ./src/get_lst_txt.py -img $1 -lbl $2 -lst $3 
python ./src/im2rec.py $3 . --num-thread 8

mkdir -p RecDataSet # elegant way, including exists checking.
mv `echo $3 | cut -d '.' -f 1`.* ./RecDataSet
echo "[INFO] RecDataSet prepared done!"
