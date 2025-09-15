% Load data (Contains X and y)
load('example_data1.mat'); % active variables: 1,2,3

% Run Indicator-based Bayesian variable selection algorithm
% X and y are required
results = Indicator_based_bayesian_vs_gp(X, y, ...
    'lambda', 1, ... % default 1
    'tau', 1, ... % default 1
    'sigma', 1, ... % default 1
    'q', 0.5, ... % default 0.5
    'iterations', 200, ... % default 2000
    'burnin', 100, ... % default 1000
    'scaling', 3, ...
    'verbose', 1);

% Display results
disp('Probabilities of active variables:');
disp(results.active_prob);

disp('Selected variables:');
disp(results.active_vars);


