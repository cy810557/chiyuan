function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  m=size(X,2);
  n = size(X, 1);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%
h = theta'*X;
f = 1/(2*m)*sum((h - y).^2);
% for j= 1: n
%     g(j)= 1/m * sum((h - y).*X(j, :), 2);
% end
g = 1/m * sum((h - y).* X, 2); %再次节省了成倍时间
