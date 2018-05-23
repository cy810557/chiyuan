 function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  groundTruth =  full(sparse(y, 1:m, 1));
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
% lambda = 1e-4;
% h = exp(theta' * X);
% p = h./sum(h,1);
% f = -1/m * (groundTruth(:)' * log(p(:))) + lambda/2 * sum(theta(:).^2);
% g = -1/m * (X * (groundTruth - p)') + lambda * theta;






  h0=exp(theta'*X);
  sum_h0=sum(h0,1);
  h=h0./sum_h0;
  for i=1:m
      for k=1:num_classes-1
          indicator=(y(i)==k);
          f=f+indicator*log(h0(k,i)/sum_h0(i));
      end
  end
  lambda  = 1e-4;
  f=-1/m*f + lambda/2 * sum(sum(theta.^2));
  
  for t=1:num_classes-1
      IDCTOR=(y==t);% logical 1*60000
      temp=X.*(IDCTOR-h(t,:)); %X:785*60000，  括号：1*60000,  想要的结果： 785*60000
      g(:,t)=g(:,t)+sum(temp,2) + lambda*theta(:,t);
  end
  g=-1/m*g;
  g=g(:); % make gradient a vector for minFunc

