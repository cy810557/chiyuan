%% 该脚本实现直方图均衡化，并与matlab自带的histeq函数进行对比
% 总结
% 1.可以使用[count,gray_level]=imhist(I) 方便地获得图像I的灰度级以及对应灰度级都像素数。
% 2. 可以使用cumsum函数直接获取矩阵的累加
% 3. 注意在循环中由于没有做到同时更新，先更新的值又参与到随后的更新中导致错误，这样一种问题
% 4. 优化：使用arrayfun()函数对所有矩阵元素使用同一个函数运算，避免循环
close all;clear;
dbstop if error
I = imread('dark.jpg');
[m, n] = size(I);
imhist(I),figure,imshow(I),title('Original Image');
[count,gray_level]=imhist(I);
gray_list = unique(sort(I(:)));
L = length(gray_list);
% num_gray = size(find(I==gray_list),1).*size(find(I==gray_list),2);
%问题：这句中的find(I==gray_list)如何用矩阵运算实现？（一种情况可以实现，所有情况一定可以用矢量运算实现,避免循环）
num_gray = zeros(L,1);
% for i = 1:L
% %     num_gray(i) = size(find(I==gray_list(i)),1)*size(find(I==gray_list(i)),2); %傻逼写法...
% num_gray(i) = length(find(I==gray_list(i)));
% end
%% 答案：使用arrayfun() 函数对所有矩阵元素使用同一个函数运算。
% f = @(i) length(find(I == gray_list(i))); %错误写法...
f = @(x) length(find(I == x));
num_gray = arrayfun(f, gray_list);
%%
probability_gray = num_gray/(m*n);
for j = 1:L
probability_accumulated(j) = sum(probability_gray(1:j)); %可以使用cumsum函数代替矩阵累加操作
end
probability_accumulated = probability_accumulated';
gray_equalized = round(probability_accumulated*256);
% 下面这段原来写的是：
% for k = 1:L
% I(find(I == gray_list(k))) = gray_equalized(k); 
% end
% 问题在于：I中某个灰度级150，被均衡之后变成158，然后I中灰度级为158的又被均衡化为新的灰度级208
% 即：I的某些值本来已被修改，但随着循环进行，再次被修改，而这是不希望看到的==>故用I1接受I的灰度值
% 这样保证I中的每个像素只被修改一次，并传给I1.
I1 = zeros(m,n);
for k = 1:L
I1(find(I == gray_list(k))) = gray_equalized(k); 
end
%
figure, imhist(uint8(I1));
figure, imshow(uint8(I1)),title('Equalized Image')
load handel;
sound(y,Fs); 
