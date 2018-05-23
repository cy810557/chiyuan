%% 
clear;clc;close all;
I1=imread('23425.png');I2=imread('23439.png');
I1=imresize(I1,[500,600]);
I2=imresize(I2,[500,600]);
%未出现负值
I_diff = imabsdiff(I2,I1); 
% I_diff = imsubtract(I2,I1); 
subplot(121),imshow(I1,[]),subplot(122),imshow(I2,[]);
 level = graythresh(I_diff);
 level1=0.09;
bw = im2bw(I_diff,level);figure,imshow(bw);title('imabsdiffUint8');

%% 可见获得的bw图像有很多sparkles。尝试下去噪：
% background = imopen(I1,strel('disk',15));figure,imshow(background);
% 注意：不能使用imnoise处理二值图像。
bw_denois=bwareaopen(bw,35); %使用形态学的函数进行下膨胀，去除bw中像素数超过35的连通域。
h1=figure,imshow(bw_denois);title('去除点状噪声之后');
%   g=imdilate(f,strel('disk',2));figure,imshow(g);
%% 计算物体运动速度：先获取像素点距离，然后找参考物体来转换
[coor1,coor2]=ginput(2); %从GUI交互式获取n个像素点的坐标（可以按enter键提前结束）。
X=[coor1';coor2'];DIS=pdist(X); %pdist()，最常用的是将所有点的坐标放进一个n*2（二维）或者n*3（三维）的矩阵X。
DIS_real = DIS*7.5/120.62;% 根据柱子间隔长度进行估计，真实距离为
delta_T = 57.057/1368;
speed=DIS_real/(delta_T*(39-25));
fprintf('HenryVIII时期重甲骑兵冲锋速度约为%d m/s\n',speed);

