% Predict value fstar at point rstar from observed values via Gaussian 
% Process Regression. Points are in a two-dimensional unti square
% domain where an m x m grid of uniformly spaced points have observations.
%
% Sample use: 
%    fstar = GPR(16,[0.5,0.5]);
%
function fstar = GPR(m, rstar)

% Initialize m x m grid of points
% XY(r,1) and XY(r,2) store the x and y coordinates of the rth point
  n = m*m;
  h = 1/(m+1);
  XY = zeros(n,2);
  idx = 0;
  for i = 1:m,
      for j = 1:m
          idx = idx+1;
          XY(idx,1) = i*h; 
          XY(idx,2) = j*h;
      end
  end
  
% Initialize observed data vector f
  f = 0.1*(rand(n,1)-0.5);
  for i = 1:n,
      f(i) = f(i) + 1.0 - (XY(i,:)-0.5)*(XY(i,:)-0.5)';
  end

% Initialize K
  K = zeros(n,n);
  for i = 1:n,
      for j = 1:n,
          d = XY(i,:)-XY(j,:);
          K(i,j) = exp(-d*d');
      end
  end
  
% Compute LU factorization of tI + K
  t = 0.01;
  [L,U] = lu(t*eye(n)+K);

% Initialize k
  k = zeros(n,1);
  for i = 1:n,
      d = rstar-XY(i,:);
      k(i) = exp(-d*d');
  end

% Compute predicted value fstar at rstar
  fstar = k'*(U\(L\f));

% Show output
  close all;
  f1 = surf(reshape(XY(:,1),m,m), reshape(XY(:,2),m,m), reshape(f,m,m));
  hold on; 
  plot3(rstar(1,1), rstar(1,2), fstar,'ro'); 

  return

