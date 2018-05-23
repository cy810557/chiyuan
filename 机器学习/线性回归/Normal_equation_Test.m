%% 这是我写的使用Normal Equation求解线性回归问题。
% 几种方法的theta比较被放在theta_set.mat中
% 注意： 记得将原始数据先进行洗牌。若不洗牌则test数据会非常分散，且准确度下降
load housing.data;
temp = housing';
temp = [ones(1, size(temp, 2)); temp];
temp = temp(:, randperm(size(temp,2)));  %randperm函数用于产生随机的整数矩阵或者向量

train.X = temp(1:end - 1, 1:400);
train.y = temp(end, 1:400);
test.X = temp(1:end - 1, 401:end);
test.y = temp(end, 401:end);
%% design_matrix —— X
X = train.X';
y = train.y' ;
theta_best = inv(X'*X)*X'*y;
actual_prices = test.y;
predicted_prices = theta_best'*test.X;
%% plot 
  [actual_prices,I] = sort(actual_prices);
  predicted_prices=predicted_prices(I);
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');