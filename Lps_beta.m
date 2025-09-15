function [ Lp ] = Lps_beta( x )
% LPS_BETA Computes the log posterior for beta
%
% Input:
%   x - Current beta values
%
% Output:
%   Lp - Log posterior value

global ModelInfo

% Extract model information
X = ModelInfo.X;        % Design matrix
y = ModelInfo.y;        % Response vector
lambda = ModelInfo.lambda;  % Hyperparameter for correlation parameter prior
tau = ModelInfo.tau;    % Hyperparameter for coefficient prior
sigma = ModelInfo.sigma;    % Variance parameter
posi = ModelInfo.posi;  % Indices of active variables
theta = ModelInfo.theta;    % Current theta values
beta = x;               % Current beta values

% Compute correlation matrix
Psi_theta = Psi(theta);

% Calculate log posterior
Lp = log(abs(det(Psi_theta))) + ...
    ((y - X(:, posi) * beta(posi)')'*(Psi_theta \ (y - X(:, posi) * beta(posi)')) + ...
    tau * sigma^2 * sum(abs(beta(posi))) + ...
    lambda * sigma^2 * sum(theta(posi))) / sigma^2;

end