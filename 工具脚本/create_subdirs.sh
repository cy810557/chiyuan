current_dir_list=(`ls -p ${1}| grep "/" | tr "\n" " "`)
sudir_list=(nosmoke_bg nosmoke_face nosmoke_susp smoke_hand smoke_nohand)
if [[ $# == 1 ]]
then 
   for i in ${current_dir_list[@]}; do for x in ${sudir_list[@]}; do mkdir ${1}/$i/$x;done;done

elif [ "${2}" = "rm" ]
then  
    echo "[+]removing predefined subdirs[+]"
	for x in ${sudir_list[@]};do ls -R ${1} | grep $x | grep ${1} | cut -d ":" -f1 | xargs -i rm -r -v {}; done
else  
    echo "[+]Usage: bash create_subdirs.sh {folder name} remove[+]"
fi
