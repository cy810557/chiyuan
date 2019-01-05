#!bin/bash

# 该脚本用于为某个文件夹中的所有文件加上由用户指定的特定前缀，若用户未指定，则默认使用文件夹名作为前缀。
# 若想对某个文件内所有行加前缀，只需: sed -i "s/^/prefix_/g" file 
# 应用场景：数据集处理时当多个不同文件夹内的图像起名较简单，如1.jpg, 2.jpg，在合并文件夹时会遇到重名。故先对不同文件夹的文件添加前缀。
# 参数说明： $1 待处理文件夹名称 $2 指定的前缀名（可选参数）$3 只对指定后缀的文件加前缀(可选参数)
#!/bin/bash

##########  argparser  ###########
until [ $# -eq 0 ]
do
  name=${1:1}; shift;
  if [[ -z "$1" || $1 == -* ]] ; then eval "export $name=true"; else eval "export $name=$1"; shift; fi  
done
##################################

if [ ${#dir_name} == 0 ]
then
    echo "[ERROR] Dir name must be specified!"
    echo "[INFO] Usage: bash add_prefix.sh -dir_name xx -prefix xx(optional) -ext xx(otional)"
    exit 1
fi

#dir_name=$dir_name;
if [ ${#prefix} == 0 ]
then
    prefix=`basename $dir_name`;    
else
    prefix=$prefix;
fi

if [ ${#ext} != 0 ]
then
    echo [INFO] file type: .$ext;
    ext=*.$ext;
else
    echo [INFO] file type: all files;
    ext=*
fi


echo [INFO] input dir_name: $dir_name
echo [INFO] add prefix: $prefix

#operate
cd $dir_name

for i in $ext;do mv $i $prefix"_"$i;done
