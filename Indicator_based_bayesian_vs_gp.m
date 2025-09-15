function results = Indicator_based_bayesian_vs_gp(X, y, varargin)
% BAYESIAN_VARIABLE_SELECTION_GP Implements the indicator-based Bayesian variable selection for GP models.
%
% Inputs:
%   X           - n×p design matrix
%   y           - n×1 vector of response values
%
% Optional Name-Value Pairs:
%   'lambda'    - Hyperparameter for correlation parameter prior (default: 1)
%   'tau'       - Hyperparameter for coefficient prior (default: 1)
%   'sigma'     - Variance parameter (default: 1)
%   'q'         - Prior probability of variable being active (default: 0.5)
%   'iterations'- Number of MCMC iterations (default: 2000)
%   'burnin'    - Number of burn-in iterations (default: 1000)
%   'scaling'   - Input scaling factor (default: 3)
%   'verbose'   - Display progress (0: none, 1: minimal, 2: detailed) (default: 1)

% Parse input arguments
p = inputParser;
addRequired(p, 'X', @isnumeric);
addRequired(p, 'y', @isnumeric);
addParameter(p, 'lambda', 1, @isnumeric);
addParameter(p, 'tau', 1, @isnumeric);
addParameter(p, 'sigma', 1, @isnumeric);
addParameter(p, 'q', 0.5, @(x) x > 0 && x < 1);
addParameter(p, 'iterations', 2000, @isnumeric);
addParameter(p, 'burnin', 1000, @isnumeric);
addParameter(p, 'scaling', 3, @isnumeric);
addParameter(p, 'verbose', 1, @isnumeric);
parse(p, X, y, varargin{:});

% Extract parameters
lambda = p.Results.lambda;
tau = p.Results.tau;
sigma = p.Results.sigma;
q = p.Results.q;
gn = p.Results.iterations;
burnin = p.Results.burnin;
scaling = p.Results.scaling;
verbose = p.Results.verbose;

% Scale inputs
X = X * scaling;

% Get dimensions
[n, p] = size(X);

% Set up global ModelInfo structure
global ModelInfo
ModelInfo = struct();
ModelInfo.X = X;
ModelInfo.y = y;
ModelInfo.n = n;
ModelInfo.p = p;
ModelInfo.lambda = lambda;
ModelInfo.tau = tau;
ModelInfo.sigma = sigma;
ModelInfo.gamma = ones(1, p);
ModelInfo.beta = ones(1, p);

% Calculate omega
omega = q/(1-q) * tau * lambda;

% Pre-allocate memory
theta = ones(gn, p);
beta  = ones(gn, p);
gamma = ones(gn+1, p);
rL1 = ones(gn, p);
rp1 = ones(gn, p);
rL0 = ones(gn, p);
rp0 = ones(gn, p);
rp = ones(gn, p);

% Patternsearch options
if verbose == 0
    display_option = 'off';
elseif verbose == 1
    display_option = 'final';
else
    display_option = 'iter';
end
options = psoptimset('Display', display_option, 'TolMesh', 0.05);

ub_t = ones(1, p) * 5;
lb_t = ones(1, p) * eps;
theta0 = ones(1, p) * 0.5;
ub_b = ones(1, p).*5;
lb_b = zeros(1, p)-5;
beta0 = ones(1, p);

if verbose > 0
    fprintf('Starting Bayesian variable selection for GP model\n');
    fprintf('Number of observations: %d, Number of variables: %d\n', n, p);
    fprintf('Parameters: lambda=%.2f, tau=%.2f, sigma=%.2f, q=%.2f\n', lambda, tau, sigma, q);
    fprintf('Running for %d iterations with %d burn-in...\n', gn, burnin);
end

% Main MCMC loop
checkna = 1; % For loop break
for k = 1:gn
    if verbose > 0 && mod(k, floor(gn/10)) == 0
        fprintf('Iteration %d of %d (%.1f%%)\n', k, gn, k/gn*100);
    end
    
    ModelInfo.posi = find(ModelInfo.gamma == 1);

    % Find theta
    maxtheta_m = ub_t; mintheta_m = lb_t; theta0_m = theta0;
    maxtheta_m(ModelInfo.gamma == 0) = eps; 
    mintheta_m(ModelInfo.gamma == 0) = eps; 
    theta0_m(ModelInfo.gamma == 0) = eps;
    [theta_m] = patternsearch(@(x) Lps_theta(x), theta0_m, [], [], [], [], mintheta_m, maxtheta_m, [], options);
    ModelInfo.theta = theta_m;
    theta(k, :) = theta_m;

    % Find beta
     maxbeta_m  = ub_b; minbeta_m  = lb_b; beta0_m = beta0;
     maxbeta_m(ModelInfo.gamma == 0) = eps; 
     minbeta_m(ModelInfo.gamma == 0) = eps; 
     beta0_m(ModelInfo.gamma == 0) = eps;
     [beta_m] = patternsearch(@(x) Lps_beta(x), beta0_m, [], [], [], [], minbeta_m, maxbeta_m, [], options);
     ModelInfo.beta = beta_m;
      beta(k, :) = beta_m;

    for l = 1:p
        % Compute p1
        ModelInfo.gamma(l) = 1;
        ModelInfo.posi = find(ModelInfo.gamma == 1);
        
        % Find theta
        maxtheta_m = ub_t; mintheta_m = lb_t; theta0_m = theta0;
        maxtheta_m(ModelInfo.gamma == 0) = eps; 
        mintheta_m(ModelInfo.gamma == 0) = eps; 
        theta0_m(ModelInfo.gamma == 0) = eps;
        [theta_m] = patternsearch(@(x) Lps_theta(x), theta0_m, [], [], [], [], mintheta_m, maxtheta_m, [], options);
        ModelInfo.theta = theta_m;

        % Find beta
         maxbeta_m  = ub_b; minbeta_m  = lb_b; beta0_m = beta0;
         maxbeta_m(ModelInfo.gamma == 0) = eps; 
         minbeta_m(ModelInfo.gamma == 0) = eps; 
         beta0_m(ModelInfo.gamma == 0) = eps;
         [beta_m, minLp1] = patternsearch(@(x) Lps_beta(x), beta0_m, [], [], [], [], minbeta_m, maxbeta_m, [], options);
         ModelInfo.beta = beta_m; 
        
        minLp1 = Lps_beta(ModelInfo.beta);
        rL1(k, l) = minLp1;
        p1 = (sqrt(sigma^2) * omega)^(sum(ModelInfo.gamma));
        rp1(k,l) = p1;

        % Compute p0
        ModelInfo.gamma(l) = 0;
        ModelInfo.posi = find(ModelInfo.gamma == 1);
        
        % Find theta
        maxtheta_m = ub_t; mintheta_m = lb_t; theta0_m = theta0;
        maxtheta_m(ModelInfo.gamma == 0) = eps; 
        mintheta_m(ModelInfo.gamma == 0) = eps; 
        theta0_m(ModelInfo.gamma == 0) = eps;
        [theta_m] = patternsearch(@(x) Lps_theta(x), theta0_m, [], [], [], [], mintheta_m, maxtheta_m, [], options);
        ModelInfo.theta = theta_m;

        % Find beta
         maxbeta_m  = ub_b; minbeta_m  = lb_b; beta0_m = beta0;
         maxbeta_m(ModelInfo.gamma == 0) = eps; minbeta_m(ModelInfo.gamma == 0) = eps; beta0_m(ModelInfo.gamma == 0) = eps;
         [beta_m, minLp0] = patternsearch(@(x) Lps_beta(x), beta0_m, [], [], [], [], minbeta_m, maxbeta_m, [], options);
         ModelInfo.beta = beta_m;

        minLp0 = Lps_beta(ModelInfo.beta);
        if isinf(minLp0)
            if minLp0 < 0
                minLp0 = -10^16;
            else
                minLp0 = 10^16;
            end
        end

        rL0(k, l) = minLp0;
        p0 = (sqrt(sigma^2) * omega)^(sum(ModelInfo.gamma)) * exp(-1/2*(minLp0-minLp1));
        rp0(k,l) = p0;
        if isinf(p0)
            p0 = 10^16;
        end

        % Compute p
        rp(k, l) = p1/(p1 + p0);
        ModelInfo.gamma(l) = binornd(1, p1/(p1 + p0), 1);

        if isnan(ModelInfo.gamma(l))
            warning('NaN encountered in gamma at iteration %d for variable %d', k, l);
            checkna = 0;
            break
        end
    end

    if checkna == 0
        warning('Breaking loop due to NaN in gamma');
        break
    end

    gamma(k+1, :) = ModelInfo.gamma;
end

% Process results
gamma_samples = gamma(burnin+2:end, :);
theta_samples = theta(burnin+1:end, :);
beta_samples = beta(burnin+1:end, :);

% Calculate probabilities of variables being active
active_prob = mean(gamma_samples, 1);

% Determine active variables (median probability criterion)
active_vars = find(active_prob >= 0.5);

% Prepare output structure
results = struct();
results.gamma_samples = gamma_samples;
results.theta_samples = theta_samples;
results.beta_samples = beta_samples;
results.active_prob = active_prob;
results.active_vars = active_vars;
results.parameters = struct('lambda', lambda, 'tau', tau, 'sigma', sigma, 'q', q);
results.ModelInfo = ModelInfo;

if verbose > 0
    fprintf('Bayesian variable selection completed.\n');
    fprintf('Identified %d active variables out of %d.\n', length(active_vars), p);
    
    if ~isempty(active_vars)
        fprintf('Active variables (indices): ');
        fprintf('%d ', active_vars);
        fprintf('\n');
        
        fprintf('Estimated probabilities: \n');
        for i = 1:length(active_vars)
            fprintf('  Variable %d: %.4f\n', active_vars(i), active_prob(active_vars(i)));
        end
    else
        fprintf('No variables were selected as active.\n');
    end
end

end
