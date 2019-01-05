#!/usr/bin/env
import os
#with open('/home/xie/AAR_Throax/patient_testing_chiyuan.INFO') as f:

with open('/home/xie/Chiyuan/export_list.INFO') as f:
    name_list = f.read().rstrip('\n').split(' ')
for target in ['rs','ims']:
    for name in name_list:
        command_str = 'get_slicenumber /home/xie/AAR_Throax/Original/{0}/{0}.IM0'.format(name)
        output_str = os.popen(command_str).read().rstrip('\n').split(' ')
        print output_str
        arg0 = int(output_str[0])
        arg1 = int(output_str[1])
        final_command_str = 'exportMath /home/xie/AAR_Throax/Recognition/NO/fixed_threshold/{0}-{1}-irfc.BIM matlab /home/xie/AAR_Throax/chiyuanOP/{0}-{1}-irfc.mat {2} {3}'.format(name,target,arg0,arg1)
        os.system(final_command_str)



