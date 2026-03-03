function [gamma, beta] = FiLMConditioning(conditioning, feature_dim)
% FiLMConditioning - Feature-wise Linear Modulation
%
% This function generates FiLM parameters (gamma and beta) for conditioning
% neural network activations on external information (e.g., target objectives).
%
% Input:
%   conditioning - Conditioning vector [n_samples, condition_dim]
%   feature_dim - Dimension of features to be modulated
%
% Output:
%   gamma - Scaling parameters [n_samples, feature_dim]
%   beta - Shifting parameters [n_samples, feature_dim]
%
% Reference:
%   Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
%   AAAI 2018

    [n_samples, condition_dim] = size(conditioning);
    
    % Simple linear projection to gamma and beta
    % In a full implementation, this would use a small neural network
    
    % Initialize projection matrices
    W_gamma = randn(condition_dim, feature_dim) * sqrt(2 / condition_dim);
    W_beta = randn(condition_dim, feature_dim) * sqrt(2 / condition_dim);
    
    b_gamma = zeros(1, feature_dim);
    b_beta = zeros(1, feature_dim);
    
    % Project conditioning to FiLM parameters
    gamma = conditioning * W_gamma + repmat(b_gamma, n_samples, 1);
    beta = conditioning * W_beta + repmat(b_beta, n_samples, 1);
    
    % Apply activation (optional)
    % gamma = sigmoid(gamma);  % Constrain to [0, 1]
    % beta = tanh(beta);       % Constrain to [-1, 1]
end

%% Sigmoid activation
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
