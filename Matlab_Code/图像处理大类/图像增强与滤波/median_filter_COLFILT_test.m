%%该脚本使用几种方法实现中值滤波，并比较结果
%% 要点1. 通过colfilt函数分块（重叠小块，或者滑动窗口）对整幅图像分快处理
% 语法：I2 = colfilt(I, [5,5], 'sliding', @max);
% 问题：比较colfilt与conv2函数的不同之处？
% 答：使用conv2对图像与模板进行卷积，只能实现线性滤波（如均值滤波或者加权均值滤波），不能实现非线性滤波。
% 而使用colfilt可以对每个滑动小块同时使用自定义的运算，可以实现整幅图像的线性或非线性滤波。
%% 要点 2. 在分块处理之前，若后续要精确处理（如使用逻辑索引），使用padarray函数预处理，从而使整幅图像的尺寸整除小块尺寸
% 语法： padInput = padarray(input,[f f],'symmetric'); 
%% 实验一：用循环的方法处理滑动窗口问题
close all; clear;
dbstop if error
I = imread('tire.tif');
figure, imshow(I,[])
I = imnoise(I, 'salt & pepper', 0.2);
figure, imshow(I,[]);
size_win = 3;
pad_I = padarray(I, [size_win, size_win]); % 这里使用padarray可以将边界处的噪声点也处理掉
% imshow(I,[])
% figure, imshow(pad_I,[])
[m, n] = size(pad_I);
step_row = 1;
step_col = 1;
block = zeros(size_win, size_win);
for i = 1:step_row :m- size_win+1
    for j = 1:step_col :n - size_win +1
        block = pad_I(i:i+2, j:j+2);
        val_median = median(block(:));
        block((size_win+1)/2, (size_win+1)/2) = val_median;
        pad_I(i:i+2, j:j+2) = block;
    end
end
denoised = pad_I(size_win+1: m-size_win, size_win + 1: n - size_win); %pad后合并
figure, imshow(denoised,[]);title('使用循环遍历各滑动窗口')
%% 实验二：使用colfilter代替循环
I_denoised = colfilt(I, [3, 3], 'sliding', @median); %注意这里的fun输入为向量即可，不需要矩阵
figure, imshow(I_denoised, []);title('使用colfilter代替循环')
%% 实验三：使用matlab自带的中值滤波
I1 = medfilt2(I, [3, 3]);
figure, imshow(I1, []); title('使用matlab自带的medfilt2去噪')


        