clear all;
%读入原始图像
I=imread('01.jpg');
subplot(1,2,1);imshow(I);
title('原始图像');
gray=rgb2gray(I);
ycbcr=rgb2ycbcr(I);%将图像转化为YCbCr空间
heighth=size(gray,1);%读取图像尺寸
width=size(gray,2);
for i=1:heighth %利用肤色模型二值化图像
    for j=1:width
        Y=ycbcr(i,j,1);
        Cb=ycbcr(i,j,2);
        Cr=ycbcr(i,j,3);
        if(Y<80 || Y>160)
            gray(i,j)=0;
        else
            if(skin(Y,Cb,Cr)==1)%根据色彩模型进行图像二值化
                gray(i,j)=255;
            else
                gray(i,j)=0;
            end
        end
    end
end
%%
% se=strel('arbitrary',eye(5));%二值图像形态学处理
% gray0=imopen(gray,se);
% subplot(1,2,2);imshowsub(gray0,gray);title('二值图像');
se2 = strel('disk', 4); 
gray2 = imopen(gray,se2);
gray3=  bwareaopen(gray2,80); 
% gray4 = imopen(gray3,strel('disk',2));

imshowsub(gray,gray2,gray3);
E = edge(gray3,'canny');

[L,num]=bwlabel(gray3,8);%采用标记方法选出图中的白色区域
stats=regionprops(L,'Eccentricity','Area','BoundingBox');%度量区域属性
A = regionprops(L,'Area');
Long = regionprops(L,'Perimeter');
Conv = regionprops(L,'ConvexArea');
eccentricities = [stats.Eccentricity]; % 该属性表征与区域具有相同标准二阶中心矩的椭圆的离心率（可作为特征）
idxOfSkittles = find(eccentricities);
statsDefects = stats(idxOfSkittles);
figure, imshow(I);
hold on;
for idx = 1 : length(idxOfSkittles)
    if  (A(idx).Area/Long(idx).Perimeter^2<0.796)
       
        h = rectangle('Position',statsDefects(idx).BoundingBox,'LineWidth',2);
        set(h,'EdgeColor',[.75 0 0]);
        hold on;
    end
end










%%


