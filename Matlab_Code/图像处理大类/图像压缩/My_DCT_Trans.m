function [B, F] = My_DCT_Trans(I)
T = dctmtx(8); %先写出DCT的变换核（固定）
% mask = [1   1   1   1   0   0   0   0  % 选择一种遮罩(masking)方式——保留直流系数和前九个交流系数           
%     1   1   1   0   0   0   0   0            %（左上角为直流系数）
%     1   1   0   0   0   0   0   0            % 注意: 未保留的高频系数即被舍弃，达到图像压缩的目的
%     1   0   0   0   0   0   0   0            % 为了观察高通滤波效果，暂未使用mask
%     0   0   0   0   0   0   0   0
%     0   0   0   0   0   0   0   0
%     0   0   0   0   0   0   0   0
%     0   0   0   0   0   0   0   0];
mask = ones(8,8); mask(4:6,4:6)=0;
fun = @(block_struct)  T * block_struct.data * T'; %DCT正变换：T*B*T' , 其中B为待变换的图像块
B = blockproc(I, [8, 8], fun); %对整幅图像I进行分块处理，同时做DCT变换
% F = B;
fun2 = @(block_struct) mask .*  block_struct.data; % 保留每个小块中指定一部分高频分量，实现压缩
F = blockproc(B, [8, 8], fun2); 
end