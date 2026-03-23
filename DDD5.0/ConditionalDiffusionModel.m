classdef ConditionalDiffusionModel < handle
% ConditionalDiffusionModel - Standard DDPM with conditional input
% Now used as a diversity-enhancing operator: given a target objective vector,
% generates a decision variable that is a perturbed version of a reference solution.
%
% Author: Refactored
% Date: 2026-03-23

    properties(SetAccess = private)
        D                      % Decision dimension
        M                      % Objective dimension
        TimeSteps              % Number of diffusion steps (T)
        Beta                   % Noise schedule variance
        Alpha                  % 1 - Beta
        AlphaBar               % Cumulative product of Alpha
        SqrtAlphaBar           % sqrt(AlphaBar)
        SqrtOneMinusAlphaBar   % sqrt(1 - AlphaBar)
        Network                % Neural network for noise prediction
        XNormalizer            % Structure with min, max for X
        FNormalizer            % Structure with min, max for F
        IsTrained = false
        UseGPU = false
        DMSteps
        Verbose = false
        EmbeddingDim
        HiddenDim
        UseDimensionalityReduction
        PCAComponents
    end
    
    methods
        function obj = ConditionalDiffusionModel(D, M, noise_schedule, dm_steps, use_gpu, verbose)
            obj.D = D;
            obj.M = M;
            obj.TimeSteps = 100;
            obj.DMSteps = min(dm_steps, obj.TimeSteps);
            obj.UseGPU = use_gpu && obj.checkDeepLearningSupport();
            obj.Verbose = verbose;
            obj.EmbeddingDim = 64;
            obj.HiddenDim = 256;
            obj.UseDimensionalityReduction = D >= 50;
            obj.PCAComponents = [];
            obj.setupNoiseSchedule(noise_schedule);
            
            obj.XNormalizer.min = -ones(1, D);
            obj.XNormalizer.max = ones(1, D);
            obj.FNormalizer.min = zeros(1, M);
            obj.FNormalizer.max = ones(1, M);
        end
        
        function hasSupport = checkDeepLearningSupport(~)
            hasSupport = false;
            try
                net = feedforwardnet([10,10]);
                hasSupport = true;
                clear net;
            catch
                hasSupport = false;
            end
        end
        
        function setupNoiseSchedule(obj, noise_schedule)
            obj.Beta = linspace(noise_schedule(1), noise_schedule(2), obj.TimeSteps);
            obj.Alpha = 1 - obj.Beta;
            obj.AlphaBar = cumprod(obj.Alpha);
            obj.SqrtAlphaBar = sqrt(obj.AlphaBar);
            obj.SqrtOneMinusAlphaBar = sqrt(1 - obj.AlphaBar);
            obj.PosteriorVar = obj.Beta .* (1 - obj.AlphaBar ./ obj.Alpha);
        end
        
        function obj = train(obj, Population, Problem, epochs)
            obj.IsTrained = false;
            if ~obj.checkDeepLearningSupport()
                warning('CDM:NoDL', 'Deep Learning Toolbox not available.');
                return;
            end
            
            try
                X = Population.decs;
                F = Population.objs;
                
                obj.XNormalizer.min = min(X, [], 1);
                obj.XNormalizer.max = max(X, [], 1);
                obj.FNormalizer.min = min(F, [], 1);
                obj.FNormalizer.max = max(F, [], 1);
                
                XNorm = obj.normalizeX(X);
                FNorm = obj.normalizeF(F);
                
                if obj.UseDimensionalityReduction && obj.D >= 50
                    [XNorm_PCA, obj.PCAComponents] = obj.applyPCA(XNorm);
                    if obj.Verbose
                        disp(['  [CDM] Applied PCA: ', num2str(obj.D), ' -> ', num2str(size(XNorm_PCA, 2)), ' dimensions']);
                    end
                else
                    XNorm_PCA = XNorm;
                end
                
                [TrainX, TrainF, TrainT, TrainNoise] = obj.prepareTrainingData(XNorm_PCA, FNorm);
                
                obj.Network = obj.buildNetwork();
                obj.Network = obj.trainNetwork(TrainX, TrainF, TrainT, TrainNoise, epochs);
                
                obj.IsTrained = true;
                if obj.Verbose
                    disp('  [CDM] Model trained successfully.');
                end
            catch ME
                warning('CDM:TrainingError', ['Training failed: ' ME.message]);
                obj.IsTrained = false;
            end
        end
        
        function [X_PCA, PCAComponents] = applyPCA(obj, X)
            n_samples = size(X, 1);
            n_features = size(X, 2);
            
            X_centered = X - mean(X, 1);
            
            if n_samples < n_features
                cov_matrix = (X_centered' * X_centered) / (n_samples - 1);
            else
                cov_matrix = (X_centered * X_centered') / (n_samples - 1);
            end
            
            if n_samples < n_features
                [~, V] = eig(cov_matrix);
                eigenvectors = X_centered' * V;
            else
                [~, eigenvectors] = eig(cov_matrix);
            end
            
            for i = 1:size(eigenvectors, 2)
                eigenvectors(:, i) = eigenvectors(:, i) / norm(eigenvectors(:, i));
            end
            
            X_PCA = X_centered * eigenvectors;
            
            explained_var = var(X_PCA, 0, 1);
            explained_var_ratio = explained_var / sum(explained_var);
            cumulative_var = cumsum(explained_var_ratio);
            
            n_components_95 = find(cumulative_var >= 0.95, 1);
            n_components_50 = min(50, size(X_PCA, 2));
            n_components = max(n_components_50, n_components_95);
            n_components = min(n_components, size(X_PCA, 2));
            
            X_PCA = X_PCA(:, 1:n_components);
            PCAComponents = struct(...
                'mean', mean(X, 1), ...
                'eigenvectors', eigenvectors(:, 1:n_components), ...
                'n_components', n_components);
        end
        
        function [TrainX, TrainF, TrainT, TrainNoise] = prepareTrainingData(obj, XNorm, FNorm)
            n_samples = size(XNorm, 1);
            
            if n_samples < 200
                aug_factor = 10;
            elseif n_samples < 500
                aug_factor = 5;
            else
                aug_factor = 3;
            end
            
            n_total = n_samples * aug_factor;
            
            TrainX = zeros(n_total, obj.D);
            TrainF = zeros(n_total, obj.M);
            TrainT = zeros(n_total, 1);
            TrainNoise = zeros(n_total, obj.D);
            
            idx = 1;
            for i = 1:n_samples
                for j = 1:aug_factor
                    t = randi(obj.TimeSteps);
                    t_norm = t / obj.TimeSteps;
                    
                    noise = randn(1, obj.D);
                    noisy_x = obj.SqrtAlphaBar(t) * XNorm(i,:) + obj.SqrtOneMinusAlphaBar(t) * noise;
                    
                    TrainX(idx,:) = noisy_x;
                    TrainF(idx,:) = FNorm(i,:);
                    TrainT(idx) = t_norm;
                    TrainNoise(idx,:) = noise;
                    
                    idx = idx + 1;
                end
            end
        end
        
        function net = buildNetwork(obj)
            input_dim = obj.D + obj.M + obj.EmbeddingDim;
            
            hidden_sizes = [obj.HiddenDim, obj.HiddenDim, obj.HiddenDim];
            
            net = feedforwardnet(hidden_sizes);
            net = configure(net, rand(input_dim, 100), rand(obj.D, 100));
            net.trainFcn = 'trainscg';
            net.trainParam.epochs = 100;
            net.trainParam.goal = 1e-6;
            net.trainParam.min_grad = 1e-7;
            net.trainParam.showWindow = false;
            net = init(net);
        end
        
        function net = trainNetwork(obj, TrainX, TrainF, TrainT, TrainNoise, epochs)
            Inputs = obj.prepareNetworkInputs(TrainX, TrainF, TrainT);
            Targets = TrainNoise';
            
            obj.Network.trainParam.epochs = epochs;
            net = train(obj.Network, Inputs, Targets);
        end
        
        function Inputs = prepareNetworkInputs(obj, X, F, T)
            n = size(X, 1);
            emb = zeros(n, obj.EmbeddingDim);
            
            for i = 1:(obj.EmbeddingDim/2)
                freq = 2 * pi * (10000^(-2*(i-1)/obj.EmbeddingDim));
                emb(:, 2*i-1) = sin(T * freq);
                emb(:, 2*i)   = cos(T * freq);
            end
            
            Inputs = [X, F, emb]';
        end
        
        function X = sample(obj, TargetObjs, ReferenceX)
            if ~obj.IsTrained
                error('Model not trained');
            end
            
            n_samples = size(TargetObjs, 1);
            FNorm = obj.normalizeF(TargetObjs);
            
            if obj.UseDimensionalityReduction && ~isempty(obj.PCAComponents)
                effective_D = obj.PCAComponents.n_components;
            else
                effective_D = obj.D;
            end
            
            if nargin >= 3 && ~isempty(ReferenceX) && size(ReferenceX,1) == n_samples
                XNorm = obj.normalizeX(ReferenceX);
                
                if obj.UseDimensionalityReduction && ~isempty(obj.PCAComponents)
                    XNorm = (XNorm - obj.PCAComponents.mean) * obj.PCAComponents.eigenvectors;
                end
                
                XNorm = XNorm + 0.1 * randn(n_samples, effective_D);
            else
                XNorm = randn(n_samples, effective_D);
            end
            
            XNorm = max(min(XNorm, 3), -3);
            
            step_size = max(1, floor(obj.TimeSteps / obj.DMSteps));
            
            for t = obj.TimeSteps:-1:step_size
                t_norm = t / obj.TimeSteps;
                t_vec = t_norm * ones(n_samples, 1);
                
                Inputs = obj.prepareNetworkInputs(XNorm, FNorm, t_vec);
                PredictedNoise = sim(obj.Network, Inputs)';
                
                if t > step_size
                    alpha_t = obj.Alpha(t);
                    alpha_bar_t = obj.AlphaBar(t);
                    beta_t = obj.Beta(t);
                    
                    coeff1 = 1 / sqrt(alpha_t);
                    coeff2 = beta_t / sqrt(1 - alpha_bar_t);
                    
                    if step_size > 1
                        alpha_bar_prev = obj.AlphaBar(max(1, t - step_size));
                        posterior_mean = coeff1 * (XNorm - coeff2 * PredictedNoise);
                        posterior_var = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t;
                        z = randn(n_samples, effective_D);
                        XNorm = posterior_mean + sqrt(posterior_var) * z;
                    else
                        sigma = sqrt(obj.PosteriorVar(t));
                        z = randn(n_samples, effective_D);
                        XNorm = coeff1 * (XNorm - coeff2 * PredictedNoise) + sigma * z;
                    end
                else
                    alpha_t = obj.Alpha(1);
                    alpha_bar_t = obj.AlphaBar(1);
                    beta_t = obj.Beta(1);
                    coeff1 = 1 / sqrt(alpha_t);
                    coeff2 = beta_t / sqrt(1 - alpha_bar_t);
                    XNorm = coeff1 * (XNorm - coeff2 * PredictedNoise);
                end
            end
            
            XNorm = max(min(XNorm, 3), -3);
            
            if obj.UseDimensionalityReduction && ~isempty(obj.PCAComponents)
                XNorm = XNorm * obj.PCAComponents.eigenvectors' + obj.PCAComponents.mean;
            end
            
            X = obj.denormalizeX(XNorm);
        end
        
        function XNorm = normalizeX(obj, X)
            range = obj.XNormalizer.max - obj.XNormalizer.min;
            range(range == 0) = 1;
            XNorm = 2 * (X - obj.XNormalizer.min) ./ range - 1;
        end
        
        function X = denormalizeX(obj, XNorm)
            range = obj.XNormalizer.max - obj.XNormalizer.min;
            range(range == 0) = 1;
            X = (XNorm + 1) / 2 .* range + obj.XNormalizer.min;
        end
        
        function FNorm = normalizeF(obj, F)
            range = obj.FNormalizer.max - obj.FNormalizer.min;
            range(range == 0) = 1;
            FNorm = (F - obj.FNormalizer.min) ./ range;
        end
    end
end