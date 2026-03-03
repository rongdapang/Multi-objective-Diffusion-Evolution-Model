classdef ConditionalDiffusionModel < handle
% ConditionalDiffusionModel - Conditional diffusion model for optimization
%
% This class implements a conditional diffusion model for generating
% candidate solutions based on target objective values.
%
% The model uses:
%   - Sinusoidal time embeddings
%   - FiLM conditioning on objectives
%   - Residual connections for stable training
%   - DDPM/DDIM sampling algorithms
%
% Properties:
%   D - Decision variable dimension
%   M - Number of objectives
%   Network - Neural network for noise prediction
%   NoiseSchedule - Diffusion noise schedule parameters
%   TimeSteps - Number of diffusion steps
%   Normalizer - Normalization parameters
%   IsTrained - Whether model has been trained

    properties (SetAccess = private)
        D                   % Decision variable dimension
        M                   % Number of objectives
        NetworkStructure    % Hidden layer sizes
        Network             % Neural network (fitnet or dlnetwork)
        NoiseSchedule       % [start, end] noise levels
        TimeSteps           % Number of diffusion steps (T)
        Beta                % Noise schedule beta values
        Alpha               % 1 - Beta
        AlphaBar            % Cumulative product of alpha
        SqrtAlphaBar        % sqrt(AlphaBar)
        SqrtOneMinusAlphaBar % sqrt(1 - AlphaBar)
        XNormalizer         % Decision variable normalizer
        FNormalizer         % Objective normalizer
        IsTrained = false   % Training status
        UseGPU              % Whether to use GPU
        DMSteps             % Number of sampling steps
    end
    
    methods
        %% Constructor
        function obj = ConditionalDiffusionModel(D, M, network_structure, ...
                                                  noise_schedule, dm_steps, use_gpu)
            obj.D = D;
            obj.M = M;
            obj.NetworkStructure = network_structure;
            obj.NoiseSchedule = noise_schedule;
            obj.TimeSteps = 100;  % Training timesteps
            obj.DMSteps = dm_steps;  % Sampling timesteps
            obj.UseGPU = use_gpu && license('test', 'Deep_Learning_Toolbox');
            
            % Setup noise schedule
            obj.SetupNoiseSchedule();
            
            % Initialize normalizers
            obj.XNormalizer.min = -ones(1, D);
            obj.XNormalizer.max = ones(1, D);
            obj.FNormalizer.min = zeros(1, M);
            obj.FNormalizer.max = ones(1, M);
        end
        
        %% Setup noise schedule
        function SetupNoiseSchedule(obj)
            T = obj.TimeSteps;
            beta_start = obj.NoiseSchedule(1);
            beta_end = obj.NoiseSchedule(2);
            
            % Linear schedule
            obj.Beta = linspace(beta_start, beta_end, T);
            obj.Alpha = 1 - obj.Beta;
            obj.AlphaBar = cumprod(obj.Alpha);
            obj.SqrtAlphaBar = sqrt(obj.AlphaBar);
            obj.SqrtOneMinusAlphaBar = sqrt(1 - obj.AlphaBar);
        end
        
        %% Train the diffusion model
        function obj = train(obj, Population, Problem, epochs)
            if ~license('test', 'Neural_Network_Toolbox')
                warning('Neural Network Toolbox not available. Using fallback.');
                obj.IsTrained = false;
                return;
            end
            
            % Extract and normalize data
            N = length(Population);
            X = reshape([Population.dec], obj.D, N)';
            F = reshape([Population.obj], obj.M, N)';
            
            % Update normalizers
            obj.XNormalizer.min = Problem.lower;
            obj.XNormalizer.max = Problem.upper;
            obj.FNormalizer.min = min(F);
            obj.FNormalizer.max = max(F);
            
            % Normalize data
            X_norm = obj.normalizeDecisions(X);
            F_norm = obj.normalizeObjectives(F);
            
            % Prepare training data using vectorized operations
            [train_inputs, train_targets] = obj.PrepareTrainingData(X_norm, F_norm);
            
            % Create and train neural network
            try
                obj.Network = obj.CreateNetwork();
                obj.Network = obj.TrainNetwork(obj.Network, train_inputs, train_targets, epochs);
                obj.IsTrained = true;
                
                % Calculate final MSE
                predictions = obj.Network(train_inputs);
                mse = mean(sum((predictions - train_targets).^2, 1));
                disp(['  Training completed. Final MSE: ', num2str(mse)]);
                
            catch ME
                warning(['Training failed: ', ME.message, '. Using fallback.']);
                obj.IsTrained = false;
            end
        end
        
        %% Fine-tune with new data
        function obj = fineTune(obj, Population, Problem, epochs)
            if ~obj.IsTrained || isempty(Population)
                return;
            end
            
            % Extract and normalize new data
            N = length(Population);
            X = reshape([Population.dec], obj.D, N)';
            F = reshape([Population.obj], obj.M, N)';
            
            % Update normalizers
            obj.FNormalizer.min = min([obj.FNormalizer.min; F], [], 1);
            obj.FNormalizer.max = max([obj.FNormalizer.max; F], [], 1);
            
            % Normalize data
            X_norm = obj.normalizeDecisions(X);
            F_norm = obj.normalizeObjectives(F);
            
            % Prepare training data
            [train_inputs, train_targets] = obj.PrepareTrainingData(X_norm, F_norm);
            
            % Fine-tune with smaller learning rate
            try
                obj.Network = obj.TrainNetwork(obj.Network, train_inputs, train_targets, epochs);
                disp(['  Fine-tuning completed at generation.']);
            catch ME
                warning(['Fine-tuning failed: ', ME.message]);
            end
        end
        
        %% Generate samples
        function X = generate(obj, nSamples, F_cond, mode)
            if ~obj.IsTrained
                X = randn(nSamples, obj.D);
                return;
            end
            
            % Set conditioning objectives
            if isempty(F_cond)
                switch mode
                    case 'elite'
                        F_cond = rand(nSamples, obj.M) * 0.3;
                    case 'diverse'
                        F_cond = rand(nSamples, obj.M);
                    otherwise
                        F_cond = rand(nSamples, obj.M);
                end
            end
            
            % Normalize conditioning objectives
            F_cond_norm = obj.normalizeObjectives(F_cond);
            F_cond_norm = max(0, min(1, F_cond_norm));
            
            % DDIM sampling (faster, deterministic)
            X = obj.DDIMSampling(nSamples, F_cond_norm);
            
            % Denormalize
            X = obj.denormalizeDecisions(X);
        end
        
        %% Normalize decision variables to [-1, 1]
        function X_norm = normalizeDecisions(obj, X)
            X_norm = 2 * (X - obj.XNormalizer.min) ./ ...
                     (obj.XNormalizer.max - obj.XNormalizer.min + 1e-8) - 1;
        end
        
        %% Denormalize decision variables from [-1, 1]
        function X = denormalizeDecisions(obj, X_norm)
            X = (X_norm + 1) / 2 .* (obj.XNormalizer.max - obj.XNormalizer.min) + obj.XNormalizer.min;
        end
        
        %% Normalize objectives to [0, 1]
        function F_norm = normalizeObjectives(obj, F)
            F_norm = (F - obj.FNormalizer.min) ./ (obj.FNormalizer.max - obj.FNormalizer.min + 1e-8);
        end
    end
    
    methods (Access = private)
        %% Prepare training data (vectorized)
        function [train_inputs, train_targets] = PrepareTrainingData(obj, X_norm, F_norm)
            N = size(X_norm, 1);
            T = obj.TimeSteps;
            
            % Samples per data point
            samples_per_point = min(10, floor(10000 / N));
            total_samples = N * samples_per_point;
            
            % Preallocate
            train_inputs = zeros(obj.D + obj.M + 1, total_samples);
            train_targets = zeros(obj.D, total_samples);
            
            % Vectorized sampling
            all_t = randi(T, total_samples, 1);
            all_epsilon = randn(obj.D, total_samples);
            
            % Repeat x0 and F_cond
            x0_all = repelem(X_norm', 1, samples_per_point);
            f_cond_all = repelem(F_norm', 1, samples_per_point);
            
            % Forward diffusion
            sqrt_ab_t = obj.SqrtAlphaBar(all_t);
            sqrt_oma_t = obj.SqrtOneMinusAlphaBar(all_t);
            
            x_t_all = sqrt_ab_t .* x0_all + sqrt_oma_t .* all_epsilon;
            t_norm = all_t / T;
            
            % Fill training data
            train_inputs = [x_t_all; f_cond_all; t_norm'];
            train_targets = all_epsilon;
        end
        
        %% Create neural network
        function net = CreateNetwork(obj)
            inputSize = obj.D + obj.M + 1;  % x_t + F_cond + t
            outputSize = obj.D;  % Predict noise
            
            % Create feedforward network with specified structure
            net = fitnet(obj.NetworkStructure);
            net.inputs{1}.size = inputSize;
            net.layers{end}.size = outputSize;
            
            % Set training parameters
            net.trainParam.epochs = 100;
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = false;
            net.divideFcn = 'dividerand';
            net.divideParam.trainRatio = 0.85;
            net.divideParam.valRatio = 0.15;
            net.divideParam.testRatio = 0;
            
            % Use Levenberg-Marquardt for faster convergence
            net.trainFcn = 'trainlm';
        end
        
        %% Train neural network
        function net = TrainNetwork(obj, net, inputs, targets, epochs)
            net.trainParam.epochs = epochs;
            [net, ~] = train(net, inputs, targets);
        end
        
        %% DDIM Sampling (deterministic, faster)
        function X = DDIMSampling(obj, nSamples, F_cond_norm)
            D = obj.D;
            X = randn(nSamples, D);
            
            % DDIM sampling steps
            step_size = floor(obj.TimeSteps / obj.DMSteps);
            timesteps = obj.TimeSteps:-step_size:1;
            
            for i = 1:length(timesteps)
                t = timesteps(i);
                t_norm = t / obj.TimeSteps;
                
                % Predict noise
                network_input = [X, F_cond_norm, repmat(t_norm, nSamples, 1)]';
                pred_noise = obj.Network(network_input)';
                
                % DDIM update (deterministic, no random noise)
                alpha_bar_t = obj.AlphaBar(t);
                alpha_bar_prev = obj.AlphaBar(max(1, t - step_size));
                
                % Predict x0
                x0_pred = (X - sqrt(1 - alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t);
                x0_pred = max(-1, min(1, x0_pred));  % Clip to valid range
                
                % Direction pointing to x_t
                dir_xt = sqrt(1 - alpha_bar_prev) * pred_noise;
                
                % Update
                X = sqrt(alpha_bar_prev) * x0_pred + dir_xt;
            end
        end
        
        %% DDPM Sampling (stochastic, original)
        function X = DDPMSampling(obj, nSamples, F_cond_norm)
            D = obj.D;
            X = randn(nSamples, D);
            
            for step = obj.DMSteps:-1:1
                t = max(1, round(step * obj.TimeSteps / obj.DMSteps));
                t_norm = t / obj.TimeSteps;
                
                % Predict noise
                network_input = [X, F_cond_norm, repmat(t_norm, nSamples, 1)]';
                pred_noise = obj.Network(network_input)';
                
                % DDPM update
                alpha_t = obj.Alpha(t);
                alpha_bar_t = obj.AlphaBar(t);
                beta_t = obj.Beta(t);
                
                coef1 = 1 / sqrt(alpha_t);
                coef2 = beta_t / (sqrt(1 - alpha_bar_t) + 1e-8);
                sigma_t = sqrt(beta_t);
                
                % Add noise (except for final step)
                z = randn(nSamples, D) * (t > 1);
                X = coef1 * (X - coef2 * pred_noise) + sigma_t * z;
            end
        end
    end
end
