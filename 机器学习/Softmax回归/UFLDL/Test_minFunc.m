%% 注意：看起来minFunc所能接收的函数fun1 必须满足有两个输出(y 以及y 的导数)。否则会出现输出数量报错
x0 = 1.5;
% options = struct('Newton', 200);
 [x,fval,exitflag,output]  = minFunc(@fun1,  x0);
 fprintf('didiu');
function [y,y_] = fun1(x)
y = (x - 2)^2;
% y_ = 2*(x-2);
y_ = 2;
end
