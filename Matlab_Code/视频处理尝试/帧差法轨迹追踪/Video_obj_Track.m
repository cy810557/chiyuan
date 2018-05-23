%% 该脚本实现对一个简单视频（桌球运动），通过帧差法和形态学处理，提取出运动轨迹
clear;close all;
disp('input video');
video = VideoReader('moving.mp4');
NumFrames = video.NumberOfFrames;
disp('output video');
% tracking(video);
testframes = read(video, [45,Inf]);
rows=size(testframes, 1);
cols=size(testframes,2);
nFrames = size(testframes,4);
testfrm = zeros(rows, cols,nFrames);
diff_frame = zeros(rows, cols,nFrames);
X=[];Y=[]; x = []; y = [];
%% 创建一个新的视频对象来接收
fps = 25;
videoName = 'tracked.avi';
newobj=VideoWriter(videoName);  %创建一个avi视频文件对象，开始时其为空
newobj.FrameRate=fps;
open(newobj); %下面写入视频之前必须先打开
for f = 1:nFrames
    testfrm(:,:,f) = (rgb2gray(testframes(:,:,:,f)));
    if f==1
        continue
    end
    diff_frame(:, :, f-1) = abs(testfrm(:,:,f)- testfrm(:,:,f-1));
    diff1 =  diff_frame(:, :, f-1);
    bw = diff_frame(:, :, f-1)>max(diff1(:)/2); 
    
    bw1 = imdilate(imerode(bw, strel('disk', 1)),strel('disk', 1)); 
    bw1 = imfill(bw1,'holes');
    bw2 = bwareaopen(bw1, 25); 
%     subplot(121),imshow(bw1),subplot(122),imshow(bw2); pause(0.1);
%  figure, imshow(bw2);
    %%
    [labeled, numObjs] = bwlabel(bw2, 8); 
    props =  regionprops(labeled, 'Centroid','Area', 'Perimeter');
    %     x=[302.0741, 313.4815];
    %     y = [192.5556, 200.5370];
    k = 1;
    for j = 1:numObjs
        if props(j).Area <25 || props(j).Area>55 || 4*pi*props(j).Area / props(j).Perimeter^2 <1.3
        continue
        else
        x(k) = props(j).Centroid(1);
        y(k) = props(j).Centroid(2);
        k = k+1;
        end
    end
    if isempty(x) 
        continue
    else
    X=[X, x];
    Y =[Y, y];
    end
    x = []; y = [];
    %  scrsz = get(0,'ScreenSize');
    % figure1=figure('Position',[0 0 scrsz(3) scrsz(4)-66]);
    figure('Visible','off'),imshow(testframes(:,:,:,f));
    hold on,  newfrm = plot(X, Y, 'MarkerSize', 8, 'Color', 'g');
    f = getframe(gca);
%     figure,imshow(f.cdata);
       writeVideo(newobj,f.cdata);
    %     figure, imshow(bw1), hold on,
    %     plot(X, Y, 'MarkerSize', 8, 'Color', 'g')
end
close(newobj);% 关闭创建视频（注意：这步不是可有可无的。否则保存的视频将打不开）