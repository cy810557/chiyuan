#在第一步标注之后，数据集中可能存在部分图像没有对应xml的情况。可以通过该脚本整理数据集，删掉没有标注的图像文件
dir=$1;
tmp_dir=Copy_of_$dir;
echo $tmp_dir;
mkdir $tmp_dir;
cp $dir/*.xml $tmp_dir;ls $tmp_dir | cut -f1 -d "." | xargs -i cp -v $dir/{}.jpg $tmp_dir;
rm -r $dir;
mv $tmp_dir $dir;
