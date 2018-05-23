%该程序实现图像目标物体的分割操作；并获得分割后区域的信息，如面积和目标数目等

%%读取图像
I=imread('Toys_Candy.jpg');
imshow(I);

%% 三通道分别通过阈值
%首先尝试直接对原rgb图像做im2bw，即 I_BW=im2bw(I,beta)；
%会发现颜色较浅的糖果也会变成白色，信息丢失。因此需通过下面的方法来分割：
% 即将图像分成rgb三个分量，分别进行二值转换，最后再求和。


% Im=double(I)/255;
r=I(:,:,1);
g=I(:,:,2);
b=I(:,:,3);

ir=im2bw(r,0.5);
ig=im2bw(g,0.5);
ib=im2bw(b,0.5);
Isum= (ir&ig&ib);  %为什么是与运算？  注意：im2bw处理之后图像类型为二值图像，即逻辑矩阵。

subplot(221);imshow(ir);title('Red Plane');
subplot(222);imshow(ig);title('Green Plane');
subplot(223);imshow(ib);title('Blue Plane');
subplot(224);imshow(Isum);title('Sum of all planes');

%%
%调用格式： IM2 = imcomplement(IM)　　函数功能： 对图像数据进行取反运算（实现底片效果）。
% 　　参数说明： IM是源图像的数据， IM2是取反后的图像数据。
% 一个简单的例子：　　X = uint8([ 255 10 75; 44 225 100]);　　X2 = imcomplement(X)
% 　　X2 = 0 245 180 211 30 155
% 注意点：　　1. 图像文件中用uint8来表示256级灰度。 对于真彩色位图， 一个像素用3个uint8分别表示该像素的R、G、B分量。
% 
% 　　2. uint8表示的数据范围： 0~255。图像的底片效果便是拿255 减去原图像数据。
% 
% ImgData = imread('poput.bmp');　　NegImgData = imcomplement(ImgData);　　figure('Name','图像的取反操作','NumberTitle','off');　　subplot(121)　　imshow(ImgData)　　title('源图像')　
% subplot(122)　　imshow(NegImgData)　　title('取反后的图像')

Icomp=imcomplement(Isum);
Ifilled=imfill(Icomp,'holes');%  imfill函数的详细用法
figure, imshow(Ifilled);



%%
%IM2 = imopen(IM,SE)
% performs morphological opening on the grayscale or binary image IM with the structuring element SE. 
% The argument SE must be a single structuring element object, as opposed to an array of objects. 
% The morphological open operation is an erosion followed by a dilation, 
% using the same structuring element for both operations.


se = strel('disk', 25);  %重要： strel（structure element）为形态学处理函数，日后学习形态学时需详细了解。
Iopenned = imopen(Ifilled,se); %形态学 打开操作，常与strel连用
figure,imshowpair(Iopenned, I);


%% Extract features
%返回区域特性 ： region propoties

Iregion = regionprops(Iopenned, 'centroid');%该函数功能强大，可以提取二值图像的多中信息
[labeled,numObjects] = bwlabel(Iopenned,4);
stats = regionprops(labeled,'Eccentricity','Area','BoundingBox');
areas = [stats.Area];
eccentricities = [stats.Eccentricity];




%% Use feature analysis to count skittles objects
idxOfSkittles = find(eccentricities);
statsDefects = stats(idxOfSkittles);

figure, imshow(I);
hold on;
for idx = 1 : length(idxOfSkittles)
        h = rectangle('Position',statsDefects(idx).BoundingBox,'LineWidth',2);
        set(h,'EdgeColor',[.75 0 0]);
        hold on;
end
if idx > 10
title(['There are ', num2str(numObjects), ' objects in the image!']);
end
hold off;




