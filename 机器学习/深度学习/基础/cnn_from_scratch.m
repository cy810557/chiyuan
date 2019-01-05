% function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
%                                 filterDim,numFilters,poolDim,pred)

%  参考：http://www.cnblogs.com/dmzhuo/p/5142438.html
%-------------------------------------------调试用参数-----------------------------------
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
   % Pooling dimension, (should divide imageDim-filterDim+1)
imageDim = 28;
% Load MNIST Train
addpath ..\rb
addpath ..\ex1
images = loadMNISTImages('..\rb\train-images.idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('..\rb\train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10
numFilters = 2;
   filterDim = 9;
    poolDim = 5;
    images = images(:,:,1:10);
    labels = labels(1:10);
% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);
%---------------------------------------------------------------------------------------------
%%
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wc为卷积层的滤波器矩阵（9,9,2）代表两个9*9的feature 
% Wd为softmax的Weighs矩阵，如(10*32)，将pooled features的obj（原来是4-D，(4,4,2,10)reshape成(32, 10)，
% 然后用Weighs与其相乘，再加上numClasses个bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
    poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
% 卷积的结果保存在activations中
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations          %保存池化之后的结果
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
% 可以直接调用cnnConvolve函数求出，也可以自己练习一遍（注意rot90，别忘啦最后的sig和bias）
% activations0 = cnnConvolve(filterDim, numFilters, images, Wc, bc);
for i = 1:numImages
    for j = 1: numFilters
        im = images(:,:,i);
        feature_filter = rot90(Wc(:,:,j),2);
        conved_im = conv2(im, feature_filter, 'valid');
        activations(:,:,j,i) = sigmoid(conved_im+bc(j));
    end
end
activationsPooled = cnnPool(poolDim, activations);
% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
% 下一步实质是将卷积池化之后的结果进行flatten，变成softmax中Data的形式
% 注意：可以循环进行卷积和池化，保证flatten之后的向量维数不至于太大
activationsPooled = reshape(activationsPooled, outputDim^2*numFilters, numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
y_hat = Wd * activationsPooled + bd;
h = exp(y_hat);  % 前向传播求出所有x(i)对应的hθ(x)。这里exp是sigmoid简化？
probs = h./sum(h, 1);  %
%第一次尝试错误：
% theta_sftmx = 0.005 * randn(numClasses*imageDim^2, 1);
% %这里softmax的theta应该是多少？重新初始化吗？
% lambda = 1e-4;
% options = struct('MaxIter', 200);
% theta(:)=minFunc(@softmax_regression_vec, theta_sftmx(:), options, activationsPooled, labels');
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.
cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
lambda = 1e-4; %正则项（惩罚）系数
groundTruth_label = full(sparse(labels, 1:numClasses, 1));
% cost  = -1/numImages * groundTruth_label(:)' * log(probs(:)) + lambda * (sum(Wd(:).^2) +sum( Wc(:).^2));
cost  = crossentropy(groundTruth_label, probs) + lambda * (sum(Wd(:).^2) +sum( Wc(:).^2));
% 尝试使用自带函数crossentropy(groundTruth_label, probs)代替-1/numImages *
% groundTruth_label(:)' * log(probs(:))  前者结果为0.2610，后者结果为0.260978

%-----------------------------------------------------------
%网上答案：
% weightDecay = 0.0001;
% logProbs = log(probs);   
% labelIndex=sub2ind(size(logProbs), labels', 1:size(logProbs,2));  %有点类似将1*10的label转换成10*10的one-hot，但是这里取各个1对应的索引
% %找出矩阵logProbs的线性索引，行由labels指定，列由1:size(logProbs,2)指定，生成线性索引返回给labelIndex
% values = logProbs(labelIndex);  
% cost = -sum(values);
% weightDecayCost = (weightDecay/2) * (sum(Wd(:) .^ 2) + sum(Wc(:) .^ 2));
% cost = cost / numImages+weightDecayCost; 


% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.
%  Use the kron function and a matrix of ones to do this upsampling
%  quickly.

%%% YOUR CODE HERE %%%
% ①SOFTMAX ERROR:
softmaxError = probs - groundTruth_label;
% ② I.卷积池化层输出误差δ(3)
%------------------------------------------------------------------------------------------------------------------------------------------------------------
%注意：
% pooledError = Wd' * softmaxError .* (activationsPooled.*(1-activationsPooled)); % %这里第一次直接用的池化后误差。
%应该将池化后误差继续反向传播经（反）池化层，变成卷积后的误差
%？？？？？？？？？？？？？？？在神经网络示意图中，这么说是不是应该用卷积的输出，而不是卷积池化的输出来“代表”hidden
%layer的神经元？？？？？？？？？？？？？？？？？？？？？
%-------------------------------------------------------------------------------------------------------------------------------------------------------------
pooledError = Wd' * softmaxError;
pooledError = reshape(pooledError, outputDim, outputDim, numFilters, numImages);
% II. 将池化之后的误差δ(2) upsample成池化前的尺寸（convolved feature map），进而求出
unpoolError = zeros(convDim, convDim, numFilters, numImages);
for i = 1:numImages
    for j = 1:numFilters
        unpoolError(:, :, j, i) = 1/poolDim^2 * kron(pooledError(:, :, j, i), ones(poolDim));
    end
end
convError = unpoolError .* activations .* (1 - activations); %理解的时候可以看成一列神经元，每个值代表每个节点的δ
%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
% softmax梯度：
Wd_grad = 1/numClasses * softmaxError * activationsPooled' + lambda * Wd;
bd_grad = 1/numClasses * sum(softmaxError, 2);
% Gradient of the convolutional layer


Wc_filter = zeros(size(Wc));
bc_grad = zeros(size(bc));
%计算bc_grad
for filterNum = 1 : numFilters
 %  δ(l)各子矩阵所有项分别求和
  bc_grad(filterNum) = (1/numImages) * sum(sum(sum(squeeze(convError(:,:, filterNum,:))))); 
end
% 计算Wc_grad
%注意，计算每个卷积核对应梯度时，要对所有example的原图与δ的卷积结果求和
for i = 1 : numFilters
    for j = 1 : numImages
        %???????????问题：为什么是旋转delta_conv，而不是旋转卷积层上一次的激励（这里即原图像）???????????
        Wc_filter(:, :, i) =Wc_filter(:, :, i) + conv2(images(:,:,j), rot90(convError(:, :, i, j), 2), 'valid');
    end
end
Wc_grad =1/numClasses * Wc_filter + lambda * Wc;
%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];
% end
