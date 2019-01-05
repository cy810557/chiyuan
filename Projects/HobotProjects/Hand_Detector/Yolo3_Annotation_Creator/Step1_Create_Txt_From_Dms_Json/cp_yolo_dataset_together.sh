# 入参：patch名称
mkdir -p YOLO_DMS/{cfg,backup,data/labels,data/images}
txt_lst=(`ls -c Outputs/${1}`)
# copy all txt files in given data patches
for x in ${txt_lst[@]};do cp Outputs/${1}/$x/*.txt YOLO_DMS/data/labels/;done
cat Outputs/train_list.txt | xargs -i cp {} YOLO_DMS/data/images/
rm train_list.txt
echo "dms_yolo dataset copied done."
