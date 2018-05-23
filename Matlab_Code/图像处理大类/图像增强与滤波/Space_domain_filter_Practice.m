%% 非线性空间滤波实例——拉普拉斯锐化进行图像增强
% 注意：imfilter返回的结果类型和输入同类，因此若输入为uint8类，输出可能会因为四舍五入丢失精度。
% 故在使用imfilter之前最好将图像转换为浮点型
% 注意：由于模板的中心系数为负数，所以最终的图像应该为：原图像减去增强后的laplacian图像。
% 可见，通过该laplacian锐化，图像的细节得到明显增强
i = imread('moon.tif');
i= im2double(i);
%% 实验一：使用中间像素为-4 的拉普拉斯锐化模板
w = fspecial('laplacian',0);
i1 = imfilter(i, w, 'replicate'); %w1为典型的拉普拉斯图像
figure, imshow(i1,[]),title('laplacian image');
img_enhanced = i - i1;
figure, imshowpair(i, img_enhanced,'montage')
%% 实验二：使用中间像素为-8的拉普拉斯锐化模板
% 问题：为啥相比-4的模板没有明显的提高？
w1 = [1 1 1 ; 1 -8 1; 1 1 1];
i2 = imfilter(i, w1, 'replicate');
figure, imshow(i2,[]),title('laplacian image');
img_enhanced2 = i - i2;
figure, imshowpair(i, img_enhanced2,'montage')
figure,imshowpair(img_enhanced,img_enhanced2,'montage')
