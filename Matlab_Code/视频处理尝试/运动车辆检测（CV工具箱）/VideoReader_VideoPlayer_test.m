%% 该脚本学习使用CVT中的视频处理函数VideoReader等，以及ForegroundDetector前景检测
% 函数，初步接触blob分析在连通域检测中的使用
% 注意，该脚本说明，CVT自带的前景预测函数具有极高的准确性
%% 注意：computer vision toolbox具有高度封装性。一种常见的使用方法是：
%%首先预定义一系列constructor，然后使用step函数把要处理的图像丢给constructor即可

% Constructor 1: VideoReader, VideoWriter, VideoPlayer等视频类object
videoReader = vision.VideoFileReader('sequence.avi');
videoPlayer = vision.VideoPlayer('Name', 'sequence');  %注意Name要大写
videoWriter = vision.VideoFileWriter('Traffic Detect.avi','FrameRate', videoReader.info.VideoFrameRate);
% Constructor 2: foregroundDetector object初始化
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 50);  
% Constructor 3 Blobanalysis 对象初始化。选择true的blob参数代表使用该方法。在本例中即使用输出
% boundingbox
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 10);
%%
% 预定义一个模板用于开操作，去除点状噪声
se = strel('disk', 2); % 注意，这里的形态学处理需要加以完善

% videoPlayer.Position(3:4) = [600 400];
count = 0;
while ~isDone(videoReader)
    count = count + 1;
    frame = step(videoReader);
    foreground = step(foregroundDetector, frame);
    foreground_filtered = imopen(foreground, se);
    bbox = step(blobAnalysis, foreground_filtered);
    result0 = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
        numCars = size(bbox, 1);
    result1 = insertText(result0, [10 10], numCars, 'BoxOpacity', 1, ...
        'FontSize', 14);
    step(videoWriter, result1);
    step(videoPlayer, result1);
end
release(videoReader);
release(videoPlayer);
release(videoWriter);