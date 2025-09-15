function [ y ] = Psi( theta )
% PSI Builds the correlation matrix for the Gaussian process model
%
% Input:
%   theta - Vector of correlation parameters
%
% Output:
%   y - Correlation matrix

global ModelInfo

n = ModelInfo.n;        % Number of observations
X = ModelInfo.X;        % Design matrix
sigma = ModelInfo.sigma;  % Variance parameter

% Pre-allocate memory for correlation matrix
Psi = zeros(n, n);

% Build upper half of correlation matrix
% Note: Commented out line suggests possibility to set theta to eps for inactive variables
% theta(gamma == 0) = eps;
for i = 1 : n
    for j = i+1 : n
        Psi(i,j) = sigma^2 * exp(-sum(theta.*abs(X(i,:) - X(j,:)).^2));
    end
end

% Complete the correlation matrix:
% 1. Add upper and lower halves (matrix is symmetric)
% 2. Add diagonal of ones multiplied by sigma^2
% 3. Add small number to diagonal to reduce ill-conditioning
y = Psi + Psi' + eye(n).*sigma^2 + eye(n).*eps;

end
