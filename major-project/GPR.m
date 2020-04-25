% Compute predictions at test points using training points and observed 
% values at those training poionts as input to the model with 
% hyperparameters t and l. 
%
function [ftest] = GPR(XY, f, itest, itrain, t, l, kernel)

  n = size(XY,1);

% Initialize K for all points (including test and training points)
 
  K0 = kernel(XY,XY,l);
  
% The kernel function above is a compact Matlab function that applies 
% to a vector of input data points and computes K0 that is equivalent to 
% the code below
%
%   K0 = zeros(n,n);
%   for i = 1:n,
%       r = XY(i,:);
%       for j = 1:n,
%           s = XY(j,:);
%           K0(i,j) = kernel(r,s,l);
%       end
%   end

% Select training and test points
  ntest = length(itest); 
  ntrain = length(itrain);
  
% Extract training set K
  K = K0(itrain,itrain);
  
% Compute LU factorization of tI + K
  [L,U] = lu(t*eye(ntrain) + K);

% Initialize k
% k(i,j) = kernel(r, s,l), where r belongs to the training set 
% and s point belong to test point set

  k = K0(itrain,itest);

% Compute predicted values
  ftest = k'*(U\(L\f(itrain)));

  return