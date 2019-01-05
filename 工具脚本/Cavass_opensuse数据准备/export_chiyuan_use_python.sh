!#/usr/bin/env
# This script tries to excute 'exportMath' with arguments acquired by pytrhon script.
patient_list = 'cat /home/xie/AAR_Throax/patient_testing_chiyuan.INFO'
for target in rps lps
do
    for name in patient_list
    do
        exportMath /home/xie/AAR_Throax/Recognition/NO/fixed_threshold/$patient_list-$target-irfc.BIM matlab /home/xie/AAR_Throax/chiyuanOP/$patient_list-$target-irfc.mat ${1} ${2}
    done
done
