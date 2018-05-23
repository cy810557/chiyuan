%% 该脚本为第十三章编程作业：①实现DCT变换 ②用DCT演示高通滤波的图像增强作用
%% 要点：
% 1.将DCT同FFT类比，可以 看作是频率域的高通滤波 
% 2.blockproc函数的使用 
% 3. DCT的实现：T *B* T'; 反DCT的实现: T' *B* T，以及通过mask舍弃高频分量实现图像压缩
% 注意：对每一小块进行高通滤波之后效果反而不明显了, 因此中间部分直接使用整幅图像DC
I = imread('cameraman.tif');
I_gray = double(I);
% imshow(I2);
[B, F]= My_DCT_Trans(I_gray);
% B = ifftshift(B);
% F = ifftshift(F);
imshowsub(log(abs(B(1:8,1:8))), log(abs(F(1:8,1:8)))); title('高通滤波前后的频谱（一小块）');
inverse_DCT = My_iDCT(F);
 figure, imshowpair(I_gray, inverse_DCT,'montage');

 %% 对整幅图像做DCT
 close all; clear;
I = imread('xu.jpg');
I= imresize(I,[480,480]);
I = imrotate(I, -90);
I_gray = im2double(rgb2gray(I));
[k,l]=size(I_gray);
T = dctmtx(480);
F = T*I_gray*T';
F = fftshift(F);
B = F; 
% B(2/5*k:3/5*k,2/5*l:3/5*l)=0; 
B(4/9*k:5/9*k,4/9*l:5/9*l)=0;
imshowsub(log(abs(F)), log(abs(B))); title('高通滤波前后的频谱');
inversed0 = T' *F * T;
inversedDCT = T' *B * T;
figure, imshow(inversed0), title('直接做DCT反变换');
figure, imshow(I_gray),title('DCT高通滤波之前');
figure, imshow(inversedDCT), title('DCT高通滤波之后')
