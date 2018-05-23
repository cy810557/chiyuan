%% 本脚本学习使用blockproc函数对矩阵进行分块处理，避免循环，可节省大量时间
fun = @(block_struct) repmat(block_struct.data,5,5) ;
%注意处理函数的输入必须是结构体，其data域是我们的矩阵数据，这是由blockproc分块后的机制决定
blockproc('moon.tif',[16 16],fun,'Destination','moonmoon.tif'); % 可以通过'Destination'这一选项，将处理的图像直接保存本地
imshow('moonmoon.tif');
%% 将图像每个小块缩小
i1 = imread('pears.png');
fun = @(block_struct)imresize(block_struct.data, 0.15);
i2 = blockproc(i1, [100 100], fun);
i3 = imresize(i1, 0.15);
imshowsub(i1, i2, i3)
%% 用每小块的标准差代替该小块像素值
i1 = imread('moon.tif');
fun2 = @(block_struct) std2(block_struct.data) * ones(size(block_struct.data)); % std2(a) 返回a的标准差（一个数）
i2 = blockproc(i1,[32,32], fun2);
imshow(i2,[])
%% 将RGB图像转换成GRB图像...
i1 = imread('peppers.png');
fun3 = @(block_struct) block_struct.data(:, :, [2 1 3]);
i2 = blockproc(i1, [200 200], fun3);
i3 = i1(:,:,[2,1,3]);
isequal(i2, i3)
imshowsub(i1, i2, i3)
%% 将超级大的图像从tiff格式转换为JPEG 2000 格式
fun4 = @(block_struct) block_struct.data;
blockproc('largeImage.tif',[1024 1024],fun4,...
   'Destination','New.jp2');
