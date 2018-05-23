%% 该脚本实现cousera上讲解的JPEG：分块—DCT—量化—iDCT
%注意：
% 1. blockproc函数处理的图像，其尺寸必须可以整除分成的小块n。否则会自动调整尺寸
% 也就是在最后的处理结果中多出来一点，以整除n
% 2. 在这里JPEG量化时为啥必须使用round?而不是向下取整？？

%% 实验1：直接反变换
close all; clear;
I = imread('road.tif');
I1 = double(rgb2gray(I));
% figure, imshow(I1,[]);
fun_dct = @(block_struct) dct2(block_struct.data);
I_dct = blockproc(I1, [8, 8], fun_dct);
% imshow(abs(log(I_dct)));
fun2 = @(block_struct) idct2(block_struct.data);
I_idct = blockproc(I_dct, [8, 8], fun2);
% figure, imshow(I_idct,[]);
error_image  = I1 - I_idct; 
imshowsub(I1, I_idct, error_image)
%% 实验2：对DCT系数进行量化
close all; clear;
I = imread('road.tif');
I1 = double(rgb2gray(I));
I1 = imresize(I1,[688, 1024]);
figure, imshow(I1,[]);
fun_dct = @(block_struct) dct2(block_struct.data);
T = blockproc(I1, [8, 8], fun_dct);
% mask=[1 1 1 1 1 1 1 1            %可用掩模来丢弃一部分高频分量
%     1 1 1 1 1 1 1 1
%     1 1 1 1 1 1 1 1
%     1 1 1 1 1 1 1 1
%     1 1 1 1 1 1 1 0
%     1 1 1 1 1 1 0 0
%     1 1 1 1 1 0 0 0
%     1 1 1 1 0 0 0 0];
% T = blockproc(T, [8, 8], @(block_struct) mask.*block_struct.data);
% imshow(abs(log(I_dct)));
% 定义量化矩阵Q
Q = [16 11 10 16 24 40 51 61
    12 12 14 19 26 58 60 55
    14 13 16 24 40 57 69 56
    14 17 22 29 51 87 80 62
    18 22 37 56 68 109 103 77
    24 35 55 64 81 104 113 92
    49 64 78 87 103 121 120 101
    72 92 95 98 112 100 103 99];
Q = 4*Q; %调整系数以实现不同程度的压缩（即压缩率）
fun_quantize = @(block_struct) round(block_struct.data./Q).*Q; %注意，这里用ceil和floor都不行？？
T_quantized = blockproc(T, [8, 8], fun_quantize); 
fun_invDCT = @(block_struct) idct2(block_struct.data);
f_idct = blockproc(T_quantized, [8, 8], fun_invDCT);
%   imshowsub(I1, f_idct);
error_image  = I1 - f_idct; 
figure, imshow(error_image,[])
imshowsub(I1, f_idct, error_image)
%% 实验3 RGB-->Ycbcr, 保持Y压缩率，增大对cb，cr通道的压缩率观察结果

