#matתnpy  
import os
import scipy.io as sio
import numpy as np
mat_dir = r'G:\Python\python_matlab_connecting'   
mat_list= os.listdir(mat_dir)
path_output = r'G:\Python\python_matlab_connecting\npy_test'
#  
for mat in mat_list:
    image = sio.loadmat(mat_dir + '/'+ mat)   
    np.save(path_output+'/' +'%s.npy' % mat[:-4], image)     
	
	
## 将numpy array保存为.mat
## sio.save('xx.mat',{'a':a,'b':b})