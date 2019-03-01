#!/bin/bash
if [ ! $# -eq 2 ]; then
    echo "[INFO] Usage: bash prepare_rec.sh {VocDataDir} {LstFileName}\n 
    For example: bash prepare_rec.sh Dataset/HandData/train hand_train.lst";
    exit 0;
fi

bash ./src/getClasses.sh $1
python ./src/get_lst_xml.py -d $1 -l $2 -c $1/ClassNames.txt
python ./src/im2rec.py $1/$2 . --num-thread 8
mkdir -p RecDataSet # elegant way, including exists checking.
mv $1/`echo $2 | cut -d '.' -f 1`.* ./RecDataSet  
echo "[INFO] RecDataSet prepared done!"
