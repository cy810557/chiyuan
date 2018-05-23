%% 用圆形来产生低通滤波器并通过信号
a=imread('cameraman.tif');
af=fftshift(fft2(a));
imshow(af);
fftshow(af);
%% 生成低通滤波器
[x, y]=meshgrid(-128:127 , -128:127);
z=sqrt(x.^ 2+y.^2);
c=z<15;
figure,imshow(c);
%% 将信号通过滤波器_空间域的卷积操作(conv2)在频率域被简化为乘积
af1=c.*af;           % 注意该语句
fftshow(af1);
iaf1=ifft2(af1);
ifftshow(iaf1);
% 振铃现象明显——是因为低通滤波器的cutoff非常尖锐
%% 增大cutoff
b=z<40;
af2=b.*af;           % 注意该语句
fftshow(af2);
iaf2=ifft2(af2);
ifftshow(iaf2);
% 总结：可用巴特沃兹滤波器消除ringing effect
