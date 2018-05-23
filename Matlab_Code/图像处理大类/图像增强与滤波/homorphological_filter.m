%% 频率域的同态滤波——most commonly used for correcting non-uniform illumination in images.
% The illumination-reflectance model of image formation: intensity at any pixel,is the product of the illumination of the scene and the reflectance(反射率) of the object(s) in the scene,i.e.
% 即—— I(x,y)=L(x,y)*R(x,y); 其中，I为图像，L为scene illumination（现场照度）； R为 scene reflectence.
% 而同态滤波用于 去除图像中拥有某些特性的乘性噪声—— we use a high-pass filter in the log domain to 
% remove the low-frequency illumination component while preserving the high-frequency reflectance component.
close all;
clear;
a=imread('trees.tif');imshow(a);
ad=im2double(a);
ad1=log(ad+0.01);
% adflog=fftshift(fft2(ad1));
adflog=fft2(ad1);
f=butterhp(a,15,1);
adfiltered=f.*adflog;
fftshow(adfiltered);
h=real(ifft2(adfiltered));%为什么取实部？？？
figure,imshow(h);
h1=exp(h);%记得用指数将对数转换回去
figure,ifftshow(h1);

