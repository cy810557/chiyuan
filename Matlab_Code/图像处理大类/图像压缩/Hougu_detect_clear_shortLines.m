%% 该脚本实现用哈尔变换滤除较小的水平边（保留较长的水平边）
I = zeros(240, 240);
I(20,30:180) = 1; I(200,80:240) = 1;
I(40,30:50) =1;  I(180,30:60) =1; I(90,140:150) =1;  
I(180:240, 200) = 1;
imshow(I);
%% 
T = [ 1 1 1 1 1 1 1 1 ;
    1 1 1 1 -1 -1 -1 -1;
    1 1 -1 -1 0 0 0 0;
    0 0 0 0  1 1 -1 -1;
    1 -1 0  0  0  0  0  0 ;
    0 0  1  -1 0  0  0  0;
    0  0  0  0  1  -1 0 0;
    0 0  0  0  0  0  1  -1]
fun = @(block_struct)  T * block_struct.data * T';
F = blockproc(I, [8, 8], fun);
imshow(F,[]);
% [LL LH HL HH]=dwt2(I, 'haar');
% I1=[LL LH;HL HH];      %一层分解
% imshow(LH,[]);
% [LL1 LH1 HL1 HH1]=dwt2(LH, 'haar');
% figure, imshow(LH1,[])
% [LL2 LH2 HL2 HH2]=dwt2(LH1, 'haar');
% figure, imshow(LH2,[])

