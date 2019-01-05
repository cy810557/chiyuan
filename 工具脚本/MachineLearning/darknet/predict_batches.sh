img_lst=(`ls -c $1`)
for x in ${img_lst[@]};
do
    img_name=`echo $x | cut -d '\' -f 4`;
    img_type=${x: -4}
    #echo $img_name
    bash detect.sh 24000 $1/$x;
    #echo predictions$img_type
    mv predictions.png pred_$img_name;
    #echo "image '$img_name' predicted....."
done
