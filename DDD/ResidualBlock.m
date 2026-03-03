function output = ResidualBlock(input, hidden_dim, activation)
% ResidualBlock - Residual connection block for neural networks
%
% This function implements a residual block with two fully connected layers
% and a skip connection.
%
% Input:
%   input - Input matrix [n_samples, input_dim]
%   hidden_dim - Hidden layer dimension
%   activation - Activation function ('relu', 'tanh', 'leaky_relu')
%
% Output:
%   output - Output matrix [n_samples, input_dim]
%
% Reference:
%   He et al. "Deep Residual Learning for Image Recognition" CVPR 2016

    if nargin < 3
        activation = 'relu';
    end
    
    [n_samples, input_dim] = size(input);
    
    % Initialize weights
    W1 = randn(input_dim, hidden_dim) * sqrt(2 / input_dim);
    W2 = randn(hidden_dim, input_dim) * sqrt(2 / hidden_dim);
    
    b1 = zeros(1, hidden_dim);
    b2 = zeros(1, input_dim);
    
    % First layer
    h = input * W1 + repmat(b1, n_samples, 1);
    h = applyActivation(h, activation);
    
    % Second layer
    h = h * W2 + repmat(b2, n_samples, 1);
    
    % Residual connection
    output = input + h;
    
    % Final activation
    output = applyActivation(output, activation);
end

%% Apply activation function
function y = applyActivation(x, activation)
    switch lower(activation)
        case 'relu'
            y = max(0, x);
        case 'leaky_relu'
            y = max(0.01 * x, x);
        case 'tanh'
            y = tanh(x);
        case 'sigmoid'
            y = 1 ./ (1 + exp(-x));
        otherwise
            y = max(0, x);  % Default to ReLU
    end
end
