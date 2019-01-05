#cd /home/users/chengyuan.yang/Pycharm_Project/hands_detector/yolo_v3
mkdir -p VOCdevkit/VOC2007/JPEGImages
mkdir -p VOCdevkit/VOC2007/ImageSets/Main
mkdir -p VOCdevkit/VOC2007/Annotations
cp Data_Factory/train_set/*.jpg VOCdevkit/VOC2007/JPEGImages/
cp Data_Factory/test_set/*.jpg VOCdevkit/VOC2007/JPEGImages/
cp Data_Factory/train_set/*.xml VOCdevkit/VOC2007/Annotations/
echo "copy image files done!"
ls Data_Factory/train_set/*.jpg >VOCdevkit/ImageSets/Main/train.txt

ls Data_Factory/test_set/*.jpg >VOCdevkit/ImageSets/Main/test.txt
echo "train.txt, test.txt created..." 
python cut_jpg.py Data_Factory/train_set/*.jpg >VOCdevkit/ImageSets/Main/train.txt
python cut_jpg.py Data_Factory/train_set/*.jpg >VOCdevkit/ImageSets/Main/test.txt
echo "start excuting voc_label.py ..."
python voc_label.py


