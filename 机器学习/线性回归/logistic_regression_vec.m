function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%
h=sigmoid(theta'*X);   % 即function set f(x)=P(c|x)=σ(∑xi*wi+b)
  for j=1:m
      f = f+ y(j)*log(1+h(j))+(1-y(j))*log(1+(1-h(j)));  %LOSS FUNC 即 Cross Entropy!
  end
  f = -f;
  g=g+X*(h-y)';
%    for i = 1: length(theta)
%       for j=1:m
%       g(i)=g(i)+X(i,j)*(h(j)-y(j));  %实际上是 f 对参数wi的微分
%       end
%   end
