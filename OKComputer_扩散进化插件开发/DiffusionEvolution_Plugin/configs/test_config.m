%% Test Configuration for DiffusionEvolution Algorithm
% This configuration file contains test parameters for quick validation
%
% Use this configuration for testing and debugging purposes.

%% Algorithm Parameters (Reduced for Testing)
params.N = 50;                           % Population size (reduced for testing)
params.diffusion_steps = 100;            % Diffusion steps (reduced for testing)
params.sample_size = 20;                 % Samples per generation (reduced)
params.hybrid_rate = 0.3;                % Diffusion offspring proportion
params.model_type = 'DDPM';              % Diffusion model type
params.training_epochs = 5;              % Training epochs (reduced for testing)
params.noise_schedule = 'linear';        % Noise schedule type
params.condition_type = 'fitness';       % Condition type

%% Advanced Parameters
params.adaptive_diffusion = true;        % Enable adaptive diffusion strength
params.memory_size = 500;                % Maximum training data size
params.diffusion_strength = 0.1;         % Base diffusion strength
params.min_hybrid_rate = 0.1;            % Minimum hybrid rate
params.max_hybrid_rate = 0.5;            % Maximum hybrid rate

%% Display configuration
fprintf('DiffusionEvolution Test Configuration:\n');
fprintf('Population Size: %d\n', params.N);
fprintf('Diffusion Steps: %d\n', params.diffusion_steps);
fprintf('Sample Size: %d\n', params.sample_size);
fprintf('Hybrid Rate: %.2f\n', params.hybrid_rate);
fprintf('Model Type: %s\n', params.model_type);
fprintf('Training Epochs: %d\n', params.training_epochs);
fprintf('Noise Schedule: %s\n', params.noise_schedule);
fprintf('Condition Type: %s\n', params.condition_type);

end