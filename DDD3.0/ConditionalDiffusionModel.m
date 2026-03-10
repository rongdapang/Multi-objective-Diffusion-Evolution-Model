classdef ConditionalDiffusionModel < handle
% ConditionalDiffusionModel - Conditional diffusion model for optimization

    properties(SetAccess = private)
        D
        M
        NetworkStructure
        Network
        NoiseSchedule
        TimeSteps
        Beta
        Alpha
        AlphaBar
        SqrtAlphaBar
        SqrtOneMinusAlphaBar
        XNormalizer
        FNormalizer
        IsTrained = false
        UseGPU
        DMSteps
        UseFallback
    end

    methods
        function obj = ConditionalDiffusionModel(D, M, network_structure, noise_schedule, dm_steps, use_gpu)
            obj.D = D;
            obj.M = M;
            obj.NetworkStructure = network_structure;
            obj.NoiseSchedule = noise_schedule;
            obj.TimeSteps = 100;
            obj.DMSteps = dm_steps;
            obj.UseFallback = ~obj.checkDeepLearningSupport();
            obj.UseGPU = use_gpu && ~obj.UseFallback;
            obj.setupNoiseSchedule();
            obj.XNormalizer.min = -ones(1, D);
            obj.XNormalizer.max = ones(1, D);
            obj.FNormalizer.min = zeros(1, M);
            obj.FNormalizer.max = ones(1, M);
        end
        
        function hasSupport = checkDeepLearningSupport(~)
            hasSupport = true;
            try
                hasSupport = license('test', 'Deep_Learning_Toolbox');
            catch
                hasSupport = false;
            end
            
            if hasSupport
                required = {'feedforwardnet', 'train', 'sim'};
                for i = 1:length(required)
                    if ~exist(required{i}, 'file')
                        hasSupport = false;
                        return;
                    end
                end
            end
        end
        
        function setupNoiseSchedule(obj)
            t = linspace(0, 1, obj.TimeSteps);
            s = 0.008;
            f_t = cos((t + s) / (1 + s) * pi / 2).^2;
            f_0 = cos(s / (1 + s) * pi / 2).^2;
            
            obj.AlphaBar = f_t / f_0;
            obj.Alpha = [obj.AlphaBar(1), obj.AlphaBar(2:end) ./ obj.AlphaBar(1:end-1)];
            obj.Beta = 1 - obj.Alpha;
            obj.SqrtAlphaBar = sqrt(obj.AlphaBar);
            obj.SqrtOneMinusAlphaBar = sqrt(1 - obj.AlphaBar);
        end
        
        function obj = train(obj, Population, Problem, epochs)
            obj.IsTrained = false;
            
            if obj.UseFallback
                warning('CDM:FallbackMode', 'Running in fallback mode - no training performed');
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
                
                [TrainX, TrainF, TrainT, TrainNoise] = obj.prepareTrainingData(XNorm, FNorm);
                
                obj.Network = obj.buildNetwork();
                obj.Network = obj.trainNetwork(TrainX, TrainF, TrainT, TrainNoise, epochs);
                
                obj.IsTrained = true;
                disp('Diffusion model trained successfully');
                
            catch ME
                warning('CDM:TrainingError', ['Training failed: ' ME.message]);
                obj.IsTrained = false;
            end
        end
        
        function [TrainX, TrainF, TrainT, TrainNoise] = prepareTrainingData(obj, XNorm, FNorm)
            n_samples = size(XNorm, 1);
            n_total = n_samples * 5;
            
            TrainX = zeros(n_total, obj.D);
            TrainF = zeros(n_total, obj.M);
            TrainT = zeros(n_total, 1);
            TrainNoise = zeros(n_total, obj.D);
            
            idx = 1;
            for i = 1:n_samples
                for j = 1:5
                    t = randi(obj.TimeSteps);
                    noise = randn(1, obj.D);
                    noisy_x = obj.SqrtAlphaBar(t) * XNorm(i, :) + obj.SqrtOneMinusAlphaBar(t) * noise;
                    
                    TrainX(idx, :) = noisy_x;
                    TrainF(idx, :) = FNorm(i, :);
                    TrainT(idx) = t / obj.TimeSteps;
                    TrainNoise(idx, :) = noise;
                    
                    idx = idx + 1;
                end
            end
        end
        
        function net = buildNetwork(obj)
            net = feedforwardnet(obj.NetworkStructure);
            net = configure(net, rand(obj.D + obj.M + 1, 10), rand(obj.D, 10));
            net.trainParam.epochs = 100;
            net.trainParam.goal = 1e-6;
            net.trainParam.min_grad = 1e-7;
            net.trainParam.showWindow = false;
            net = init(net);
        end
        
        function net = trainNetwork(obj, TrainX, TrainF, TrainT, TrainNoise, epochs)
            Inputs = [TrainX, TrainF, TrainT]';
            Targets = TrainNoise';
            
            obj.Network.trainParam.epochs = epochs;
            net = train(obj.Network, Inputs, Targets);
        end
        
        function X = sample(obj, TargetObjs)
            if ~obj.IsTrained
                error('Model not trained');
            end
            
            n_samples = size(TargetObjs, 1);
            FNorm = obj.normalizeF(TargetObjs);
            XNorm = randn(n_samples, obj.D);
            
            timesteps = linspace(obj.TimeSteps, 1, obj.DMSteps);
            
            for i = 1:length(timesteps)
                t = round(timesteps(i));
                t_norm = t / obj.TimeSteps * ones(n_samples, 1);
                Inputs = [XNorm, FNorm, t_norm]';
                PredictedNoise = sim(obj.Network, Inputs)';
                
                if i < length(timesteps)
                    t_next = round(timesteps(i+1));
                    alpha_t = obj.AlphaBar(t);
                    alpha_next = obj.AlphaBar(max(t_next, 1));
                    x_0 = (XNorm - sqrt(1 - alpha_t) * PredictedNoise) / sqrt(alpha_t);
                    XNorm = sqrt(alpha_next) * x_0 + sqrt(1 - alpha_next) * PredictedNoise;
                else
                    alpha_t = obj.AlphaBar(t);
                    XNorm = (XNorm - sqrt(1 - alpha_t) * PredictedNoise) / sqrt(alpha_t);
                end
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
            X = (XNorm + 1) / 2 .* range + obj.XNormalizer.min;
        end
        
        function FNorm = normalizeF(obj, F)
            range = obj.FNormalizer.max - obj.FNormalizer.min;
            range(range == 0) = 1;
            FNorm = (F - obj.FNormalizer.min) ./ range;
        end
    end
end
