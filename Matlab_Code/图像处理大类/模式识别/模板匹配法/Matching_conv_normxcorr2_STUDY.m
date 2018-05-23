%% 该脚本研究特征匹配的细节——分别使用互相关和卷积的方法
% 小问题：imshow很小的矩阵时，显示结果很小。如何放大？（简单起见可以用subplot）
n = 11;  m = 3;
A=eye(n);
B=A(:,end:-1:1);
C=A+B; C((n+1)/2, (n+1)/2)=1;
a = eye(m);
b = a(:, end:-1:1);
c = a+b;c((m+1)/2, (m+1)/2)=1;
subplot(121),imshow(c),title('模板(3x3)'), subplot(122), imshow(C),title('原图(11x11)');
%% 实验一：使用自带函数求归一化互相关函数——normxcorr2:
cross_corr0 = normxcorr2(c, C);
cc_abs = abs(cross_corr0);
imshowsub(c,C,cc_abs);title('归一化互相关矩阵');
% 通过观察可知，互相关矩阵值为1时为最佳匹配点，可以看成是匹配中心坐标。
[x_peak, y_peak] = find(cc_abs == max(cc_abs(:)));
x = x_peak - (m-1);
y = y_peak - (m-1);
imshow(C),hold on;
rectangle(gca,'Position',[x, y, m,m],'EdgeColor','r');
%% 实验二使用老师的方法——傅里叶变换求归一化互相关函数
% 观察结论：当u=5,v=5 即搜索的小块和模板一致时，fc_abs矩阵最大值为5，应该是所有fc_abs中最大的值，但是
% 此时fc_abs矩阵中其他点像素值并不大。
% 因此，对于每个u, v ，不应该将fc_abs矩阵的和代表该点互相关函数，而应该使用fc_abs矩阵最大值来代表
I1 = c;
I2  = C;
I1(m+m-1, m+m-1) = 0; %注意：求(m+m-1)点fft
F1 = fftshift(fft2(I1));
search_step = 1;
for u = 1: search_step: n-m+1
    for v = 1: search_step: n-m+1
        % for u = 5
        %     for v = 5
        % for u = 1
        %     for v = 1
       % Fourier Transform
        I2_patch = I2(u:u+m-1, v:v+m-1);
        I2_patch(m+m-1, m+m-1) = 0; %注意观察u = 5, v = 5的小块
        F2 = fftshift(fft2(I2_patch));
        %fftshow(F1), fftshow(F2);
        Fc = conj(F1).* F2;  %为什么是点乘
        fc = ifft2(Fc);
        fc_abs = abs(fc);
        %         调试时使用：
        %         imshowsub(I1, I2_patch, fc_abs);imshow(fc_abs,[]);
        %         subplot(121), imshow(I2_patch);subplot(122), imshow(I1),  title(['u:',num2str(u),'   v:', num2str(v)]);
        %         pause(0.2);
        Cross_corr= max(fc_abs(:));
        sum_patch = sum(sum(I2_patch));  %#############注意：可能有些小块全黑，像素值之和为零。故考虑设置阈值
        if sum_patch <1
            Measure(u, v) = 0;
        else
            Measure(u, v) = Cross_corr/sqrt(sum_patch);
        end
        clear Cross_corr ave_patch;
    end
end
[x1, y1] = find(Measure == max(Measure(:))); %注意：find函数不仅可以返回单下标索引，还可以返回多下标索引！！
imshow(C);
rectangle('Position',[x1, y1, m, m],'EdgeColor','r');

%% 实验三：尝试使用卷积一：整体卷积 (但是整体卷积无法用于合照的处理，应该是测度没考虑图像亮度)
close all;
% conv = conv2(C, c);
% subplot(121),imshow(C),subplot(122),imshow(conv,[]);
% for i = 1 : 10
%     conv = conv2(conv, c, 'same');
%     subplot(121),imshow(C),subplot(122),imshow(conv,[]); pause(0.5); title(['第',num2str(i),'层卷积...']);%似乎没什么效果
% end
conv_Cc = conv2(C, c ,'same');
[conv_x, conv_y] = find(conv_Cc == max(conv_Cc(:)));
x_m3 = conv_x - .5*(m+1); % method_3
y_m3 = conv_y - .5*(m+1);
subplot(121), imshow(C), subplot(122), imshow(C), hold on;
rectangle('Position',[x_m3, y_m3, m, m],'EdgeColor','r');
%%  实验四：尝试使用卷积二：小块间卷积
% I1 = c;
% I2  = C;
% search_step = 1;
%  for u = 1: search_step: n-m+1
%      for v = 1: search_step: n-m+1
% for u = 5
%     for v = 5
%                 I2_patch = I2(u:u+m-1, v:v+m-1);
%                 conv0 = conv2(I2_patch, I1);
%                 subplot(131),imshow(I1),subplot(132),imshow(I2_patch);
%                 subplot(133), imshow(conv0, []); pause(0.5);
%     end
% end
I1 = c;
I2  = C;
search_step = 1;
for u = 1: search_step: n-m+1
    for v = 1: search_step: n-m+1
        I2_patch = I2(u:u+m-1, v:v+m-1);
        conv_matrix2 = conv2(I1, I2_patch, 'same');
        % 测度：
        conv_max= max(conv_matrix2(:));
        sum_patch = sum(sum(I2_patch));  %#############注意：可能有些小块全黑，像素值之和为零。故考虑设置阈值
        Measure(u, v) = conv_max/sqrt(sum_patch);
        clear conv_max ave_patch;
    end
end
[x1, y1] = find(Measure == max(Measure(:))); %注意：find函数不仅可以返回单下标索引，还可以返回多下标索引！！
imshow(C);
rectangle('Position',[x1, y1, m, m],'EdgeColor','r');


