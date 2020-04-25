% Driver file to determine hyperparameters for GPR
% by using grid search in the parameter space to  
% minimize mean square error at test points that
% are picked randomly from the data; the remaining 
% points are training points

% ----------------------------------------------------------------------
% Function Deffinitions

% GPR kernel function: square exponential (applies to a vector of points)
% To be used in function GPR
%
% For two points x and y in 2D space, 
%    kernel = @(x,y,l)exp(-(1/2)*(x-y)*[1/l(1), 0; 0, 1/l(2)]*(x-y)');
%
% When x is a vector of n points, i.e.,
%    x(i,:) = [x-coordinate of x_i, y-coord of x_i]
%
% the kernel function below returns an nxn matrix K s.t.
%    K(i,j) = kernel(x(i,:),x(j,:),l)
%
% Below is a compact definition specifically for Matlab
kernel = @(x,y,l) ...
           exp(-(1/2)* ...
               ((x(:,1)* ones(1,size(y,1))-ones(size(x,1),1)*y(:,1)').^2/(l(1).^2) ...
               +(x(:,2)* ones(1,size(y,1))-ones(size(x,1),1)*y(:,2)').^2/(l(2).^2)));

% ----------------------------------------------------------------------
% Main program

% Reset random number generator to ensure reproducibility
  rng('default');
  
% *** The data points grid defined next SHOULD NOT BE CONFUSED WITH 
% *** the grid for the parameter space which will be searched to 
% *** determine the optimum values of the parametes.

% Initialize 2D grid whose points are used as data points
%   m:  generate m x m grid of data points
%   XY: XY(r,1) and XY(r,2) store the x and y coordinates of the rth point
%
% m can be increased to genearate larger data sets
m = 32; 

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
  
% Examples of f (uncomment the one to use)
  f = 0.02*(rand(n,1)-0.5);
  f = f + kernel(XY, [0.25,0.25], [2; 2]/m) + XY * [0.2; 0.1];
  
% Plot f
  close all;
  f1 = figure; ax = gca;
  surf(ax,reshape(XY(:,1),m,m), reshape(XY(:,2),m,m), reshape(f,m,m));


  
% Compute hyperparameters
%
% For the code below, it does not matter how the data was generated or
% where the data came from, as long as XY has the corrdinates of data 
% points in 2D space and f has the oberved value
% Initialization of Tparam and Lparam to "appropriate" ranges, however,
% is necessary to get useful values of hyperparameters.
% Also, data should be such that an accurate GPR model exists, i.e., one
% can find hyperparameters to represent the data with a high degree of 
% accuracy.

% Select 10% points as test point randomly 
% and mark the remaining 90% as training points
  ntest = round(0.1*n);         % No. of test points
  ntrain = n - ntest;           % No. of training points
  rperm = randperm(n);
  itest = rperm(1:ntest);       % Indices of test points (randomly chosen)
  itrain = rperm(ntest+1:n);    % Indices of training points
  
% Compute mse at a grid of points in the parameter space
% Parameters l1 and l2 are assigned values from Lparam
% Parameter t assigned values from Tparam 
  Tparam = [0.5:0.5:0.5];        % Select based on data
  Lparam = [0.25:0.5:10]/m;     % Select based on data
  
  % NOTE: For simplicity, in the code below, we have decided to 
  % search the parameter space for l1 and l2 only; t remains fixed.
  
  % MSE at each grid point in parameter space
  MSE = zeros(length(Lparam),length(Lparam));
  fprintf("Begining now ...\n"); 
  it = 1;  % Fix t to be Tparam(1)
  for il1 = 1:length(Lparam),      
      for il2 = 1:length(Lparam),
          ftest = GPR(XY, f, itest, itrain, Tparam(it), [Lparam(il1); Lparam(il2)], kernel);
          error = f(itest) - ftest;
          MSE(il1,il2) = error'*error;
          fprintf("Finished (l1,l2) = %f, %f, mse = %e\n", ...
              Lparam(il1), Lparam(il2), MSE(il1,il2));
      end
  end
  
% Show output - plot log(MSE) to enhance the difference
  f2 = figure; ax = gca; hold on; 
  title('log(MSE)'); xlabel('l_1'); ylabel('l_2');
  contourf(ax,Lparam,Lparam,log(MSE));

  colorbar;
  

