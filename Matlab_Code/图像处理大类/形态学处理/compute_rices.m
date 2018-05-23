%%计算rice.png中有几颗米粒

I=imread('rice.png');
%% 自动获得阈值，转为二值图像
level1=graythresh(I);
I1=im2bw(I,level1);
imshowpair(I,I1);

%% 去除背景
background=imopen(I,strel('disk',15));imshow(background);
I2=I-background; %I2=imsubtract(I,background);
imshowpair(I,I2)

%% 对去除背景的图像转为二值图像
level2=graythresh(I2);I3=im2bw(I2,level2);
imshowpair(I,I3);%可见，图像中仍有少许噪声（sparkle），影响米粒的计数考虑 使用简单的中值滤波去噪

%% 两种滤波方式比较
 h=[0 0.1 0;0.1 0.1 0.1 ;0 0.1 0];
K=imfilter(I3,h);imshowpair(I,K);

%% 再对滤波后的图像锐化一下
K1=imfilter(K,[0 -1 0;-1 5 -1 ;0 -1 0]);imshowpair(K,K1);% 好像没什么效果..

%% connected-component labeling：对图像中的连通区域重新编号并计数
[labeled, numObjects]=bwlabel(K,8);
stats=regionprops(labeled,'BoundingBox');
for i=1:numObjects
    box=stats(i).BoundingBox;
%     x=box(1);%矩形坐标X
%     y=box(2);%矩形坐标Y
%     w=box(3);%矩形宽度w
%     h=box(4);%矩形高度h
     rectangle('Position',[box(1,1),box(1,2),box(1,3),box(1,4)],'EdgeColor','r');
end

%% 计算最大米粒的尺寸和平均尺寸
% 其实这里可以用find()更快一些
[m,n]=size(I);
num=zeros(numObjects,1);
for i=1:m
    for j=1:n
        for k=1:numObjects
            if (labeled(i,j)==k)
                num(k)=num(k)+1;
            end
        end
    end
end
% 问题：有两个相连的米粒被计数为了一个，怎么解决？ 
% 原因： 中值滤波后图像产生了模糊.将滤波器的数值调小可以分开相连的米粒，但是会导致图中最小的米粒丢失，并且会使第三颗米粒断成两颗
% 总结：图像锐化的使用尚未掌握。
%其实可以不用中值滤波，而采用高低两个阈值来滤除过大或者过小的米粒

    