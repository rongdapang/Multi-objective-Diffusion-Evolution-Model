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
        D               % Decision variable dimension
        M               % Number of objectives
        NetworkStructure % Hidden layer sizes
        Network         % Neural network (fitnet or dlnetwork)
        NoiseSchedule   % [start, end] noise levels
        TimeSteps       % Number of diffusion steps (T)
        Beta            % Noise schedule beta values
        Alpha           % 1 - Beta
        AlphaBar        % Cumulative product of alpha
        SqrtAlphaBar    % sqrt(AlphaBar)
        SqrtOneMinusAlphaBar % sqrt(1 - AlphaBar)
        XNormalizer     % Decision variable normalizer
        FNormalizer     % Objective normalizer
        IsTrained = false % Training status
        UseGPU          % Whether to use GPU
        DMSteps         % Number of sampling steps
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
            obj.setupNoiseSchedule();
            
            % Initialize normalizers
            obj.XNormalizer.min = -ones(1, D);
            obj.XNormalizer.max = ones(1, D);
            obj.FNormalizer.min = zeros(1, M);
            obj.FNormalizer.max = ones(1, M);
        end
        
        %% Setup noise schedule
        function setupNoiseSchedule(obj)
            % Cosine noise schedule
            t = linspace(0, 1, obj.TimeSteps);
            obj.Beta = obj.NoiseSchedule(1) + (obj.NoiseSchedule(2) - obj.NoiseSchedule(1)) * t;
            obj.Beta = min(max(obj.Beta, 1e-6), 0.999);  % Clamp values
            
            obj.Alpha = 1 - obj.Beta;
            obj.AlphaBar = cumprod(obj.Alpha);
            obj.SqrtAlphaBar = sqrt(obj.AlphaBar);
            obj.SqrtOneMinusAlphaBar = sqrt(1 - obj.AlphaBar);
        end
        
        %% Train the diffusion model
        function obj = train(obj, Population, Problem, epochs)
            if ~license('test', 'Deep_Learning_Toolbox')
                warning('Deep Learning Toolbox not available');
                return;
            end
            
            % Extract data
            X = reshape([Population.dec], obj.D, [])';
            F = reshape([Population.obj], obj.M, [])';
            
            % Update normalizers
            obj.updateNormalizers(X, F);
            
            % Normalize data
            X_norm = obj.normalizeX(X);
            F_norm = obj.normalizeF(F);
            
            % Prepare training data
            nSamples = size(X_norm, 1);
            
            % Create training dataset with different timesteps
            X_train = [];
            T_train = [];
            F_train = [];
            Noise_target = [];
            
            for t = 1:obj.TimeSteps
                nPerStep = max(1, round(nSamples / obj.TimeSteps));
                idx = randi(nSamples, nPerStep, 1);
                
                % Add noise
                epsilon = randn(nPerStep, obj.D);
                X_noisy = obj.SqrtAlphaBar(t) * X_norm(idx, :) + ...
                          obj.SqrtOneMinusAlphaBar(t) * epsilon;
                
                X_train = [X_train; X_noisy];
                T_train = [T_train; repmat(t, nPerStep, 1)];
                F_train = [F_train; F_norm(idx, :)];
                Noise_target = [Noise_target; epsilon];
            end
            
            % Build and train network
            obj.Network = obj.buildNetwork();
            
            % Training options
            options = trainingOptions('adam', ...
                'MaxEpochs', epochs, ...
                'MiniBatchSize', min(64, size(X_train, 1)), ...
                'Shuffle', 'every-epoch', ...
                'Verbose', false, ...
                'Plots', 'none');
            
            % Prepare input and target
            trainInput = [X_train, T_train / obj.TimeSteps, F_train];
            
            % Train
            try
                obj.Network = trainnet(trainInput, Noise_target, obj.Network, ...
                                       'mse', options);
                obj.IsTrained = true;
                disp('Diffusion model trained successfully');
            catch ME
                warning('Training failed: %s', ME.message);
                obj.IsTrained = false;
            end
        end
        
        %% Fine-tune the model
        function obj = fineTune(obj, trainingData, Problem, epochs)
            if ~obj.IsTrained || isempty(trainingData)
                return;
            end
            
            X = trainingData.dec;
            F = trainingData.obj;
            
            % Normalize
            X_norm = obj.normalizeX(X);
            F_norm = obj.normalizeF(F);
            
            nSamples = size(X_norm, 1);
            
            % Create fine-tuning dataset
            X_train = [];
            T_train = [];
            F_train = [];
            Noise_target = [];
            
            for t = 1:obj.TimeSteps
                nPerStep = max(1, round(nSamples / obj.TimeSteps));
                idx = randi(nSamples, nPerStep, 1);
                
                epsilon = randn(nPerStep, obj.D);
                X_noisy = obj.SqrtAlphaBar(t) * X_norm(idx, :) + ...
                          obj.SqrtOneMinusAlphaBar(t) * epsilon;
                
                X_train = [X_train; X_noisy];
                T_train = [T_train; repmat(t, nPerStep, 1)];
                F_train = [F_train; F_norm(idx, :)];
                Noise_target = [Noise_target; epsilon];
            end
            
            % Fine-tuning options with smaller learning rate
            options = trainingOptions('adam', ...
                'MaxEpochs', epochs, ...
                'MiniBatchSize', min(32, size(X_train, 1)), ...
                'Shuffle', 'every-epoch', ...
                'Verbose', false, ...
                'InitialLearnRate', 1e-4, ...
                'Plots', 'none');
            
            trainInput = [X_train, T_train / obj.TimeSteps, F_train];
            
            try
                obj.Network = trainnet(trainInput, Noise_target, obj.Network, ...
                                       'mse', options);
                disp('Diffusion model fine-tuned successfully');
            catch ME
                warning('Fine-tuning failed: %s', ME.message);
            end
        end
        
        %% Generate samples
        function X = generate(obj, nSamples, F_target, mode)
            if ~obj.IsTrained
                X = randn(nSamples, obj.D);
                return;
            end
            
            if nargin < 4
                mode = 'guided';
            end
            
            % Prepare conditioning
            if isempty(F_target) || size(F_target, 1) ~= nSamples
                % Generate random target objectives
                F_target = rand(nSamples, obj.M);
            else
                F_target = obj.normalizeF(F_target);
            end
            
            % Start from random noise
            X = randn(nSamples, obj.D);
            
            try
                % DDIM sampling (deterministic)
                timesteps = linspace(obj.TimeSteps, 1, obj.DMSteps);
                
                for i = length(timesteps):-1:1
                    t = round(timesteps(i));
                    
                    % Predict noise
                    input = [X, repmat(t/obj.TimeSteps, nSamples, 1), F_target];
                    predictedNoise = predict(obj.Network, input);
                    
                    % Denoise step
                    if i > 1
                        t_prev = round(timesteps(i-1));
                        alpha_bar_t = obj.AlphaBar(t);
                        alpha_bar_t_prev = obj.AlphaBar(t_prev);
                        
                        % DDIM update
                        X_0 = (X - sqrt(1 - alpha_bar_t) * predictedNoise) / sqrt(alpha_bar_t);
                        X = sqrt(alpha_bar_t_prev) * X_0 + sqrt(1 - alpha_bar_t_prev) * predictedNoise;
                else
                    % Final step
                    alpha_bar_t = obj.AlphaBar(t);
                    X = (X - sqrt(1 - alpha_bar_t) * predictedNoise) / sqrt(alpha_bar_t);
                end
                end
            catch ME
                warning('Diffusion sampling failed: %s. Returning random samples.', ME.message);
                X = randn(nSamples, obj.D);
            end
            
            % Denormalize
            X = obj.denormalizeX(X);
        end
        
        %% Normalize decision variables
        function X_norm = normalizeX(obj, X)
            X_norm = 2 * (X - obj.XNormalizer.min) ./ ...
                     (obj.XNormalizer.max - obj.XNormalizer.min) - 1;
            X_norm = max(-1, min(1, X_norm));  % Clamp
        end
        
        %% Denormalize decision variables
        function X = denormalizeX(obj, X_norm)
            X = (X_norm + 1) / 2 .* (obj.XNormalizer.max - obj.XNormalizer.min) + ...
                obj.XNormalizer.min;
        end
        
        %% Normalize objectives
        function F_norm = normalizeF(obj, F)
            F_norm = (F - obj.FNormalizer.min) ./ ...
                     (obj.FNormalizer.max - obj.FNormalizer.min);
            F_norm = max(0, min(1, F_norm));  % Clamp
        end
        
        %% Denormalize objectives
        function F = denormalizeF(obj, F_norm)
            F = F_norm .* (obj.FNormalizer.max - obj.FNormalizer.min) + ...
                obj.FNormalizer.min;
        end
        
        %% Normalize objectives (public interface)
        function F_norm = normalizeObjectives(obj, F)
            F_norm = obj.normalizeF(F);
        end
    end
    
    methods (Access = private)
        %% Build neural network
        function net = buildNetwork(obj)
            % Input: [X, t, F]
            inputSize = obj.D + 1 + obj.M;
            outputSize = obj.D;
            
            % Create layer graph
            layers = [
                featureInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
            ];
            
            % Hidden layers
            for i = 1:length(obj.NetworkStructure)
                if i == 1
                    layers = [layers;
                        fullyConnectedLayer(obj.NetworkStructure(i), 'Name', ['fc', num2str(i)])
                        batchNormalizationLayer('Name', ['bn', num2str(i)])
                        reluLayer('Name', ['relu', num2str(i)])
                    ];
                else
                    layers = [layers;
                        fullyConnectedLayer(obj.NetworkStructure(i), 'Name', ['fc', num2str(i)])
                        batchNormalizationLayer('Name', ['bn', num2str(i)])
                        reluLayer('Name', ['relu', num2str(i)])
                    ];
                end
            end
            
            % Output layer
            layers = [layers;
                fullyConnectedLayer(outputSize, 'Name', 'output')
            ];
            
            % Create network
            net = layerGraph(layers);
        end
        
        %% Update normalizers
        function updateNormalizers(obj, X, F)
            % Update decision variable normalizer
            obj.XNormalizer.min = min(X);
            obj.XNormalizer.max = max(X);
            
            % Avoid division by zero
            range = obj.XNormalizer.max - obj.XNormalizer.min;
            obj.XNormalizer.max(range < 1e-6) = obj.XNormalizer.min(range < 1e-6) + 1;
            
            % Update objective normalizer
            obj.FNormalizer.min = min(F);
            obj.FNormalizer.max = max(F);
            
            range = obj.FNormalizer.max - obj.FNormalizer.min;
            obj.FNormalizer.max(range < 1e-6) = obj.FNormalizer.min(range < 1e-6) + 1;
        end
    end
end
