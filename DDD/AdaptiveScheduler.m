classdef AdaptiveScheduler < handle
% AdaptiveScheduler - Adaptive scheduling for DM/GA contribution ratio
%
% This class implements an adaptive scheduler that dynamically adjusts
% the ratio of diffusion model offspring based on:
%   - Current generation (annealing schedule)
%   - DM offspring quality (survival rate)
%   - Optimization stagnation detection
%
% The scheduler follows a phased approach:
%   Phase 1 (Gen 1-5): Pure GA (0-5% DM)
%   Phase 2 (Gen 6-15): Warm-up (5-25% DM)
%   Phase 3 (Gen 16-50): Steady-state (25-40% DM)
%   Phase 4 (Gen 51+): Adaptive (20-50% DM)
%
% Properties:
%   MaxDMRatio - Maximum DM contribution ratio
%   WarmupGenerations - Generations for warm-up phase
%   SteadyStateGenerations - Generations for steady-state phase
%   MinDMRatio - Minimum DM contribution ratio
%
% Methods:
%   getDMRatio(generation, metrics) - Calculate current DM ratio

    properties (SetAccess = private)
        MaxDMRatio          % Maximum DM contribution ratio
        WarmupGenerations   % Generations for warm-up phase
        SteadyStateGenerations % Generations for steady-state phase
        MinDMRatio          % Minimum DM contribution ratio
        QualityHistory      % History of DM quality
    end
    
    methods
        %% Constructor
        function obj = AdaptiveScheduler(max_dm_ratio, warmup_gens, steady_state_gens)
            obj.MaxDMRatio = max_dm_ratio;
            obj.WarmupGenerations = warmup_gens;
            obj.SteadyStateGenerations = steady_state_gens;
            obj.MinDMRatio = 0.1;
            obj.QualityHistory = [];
        end
        
        %% Get adaptive DM ratio
        function ratio = getDMRatio(obj, generation, metrics)
            % Base annealing curve based on generation
            baseRatio = obj.getBaseRatio(generation);
            
            % Quality-based modulation
            qualityFactor = obj.getQualityFactor(metrics);
            
            % Stagnation-based reduction
            stagnationFactor = obj.getStagnationFactor(metrics);
            
            % Exploration factor (increase diversity when stuck)
            explorationFactor = obj.getExplorationFactor(metrics);
            
            % Combine factors
            ratio = baseRatio * qualityFactor * stagnationFactor * explorationFactor;
            
            % Clamp to valid range
            ratio = max(obj.MinDMRatio, min(obj.MaxDMRatio, ratio));
            
            % Update quality history
            if isfield(metrics, 'dmSurvivalRate')
                obj.QualityHistory = [obj.QualityHistory, metrics.dmSurvivalRate];
                if length(obj.QualityHistory) > 20
                    obj.QualityHistory = obj.QualityHistory(end-19:end);
                end
            end
        end
    end
    
    methods (Access = private)
        %% Base annealing ratio
        function ratio = getBaseRatio(obj, generation)
            if generation <= 5
                % Phase 1: Pure GA establishment
                ratio = obj.MaxDMRatio * 0.1 * (generation / 5);
            elseif generation <= obj.WarmupGenerations
                % Phase 2: Warm-up
                progress = (generation - 5) / (obj.WarmupGenerations - 5);
                ratio = obj.MaxDMRatio * (0.1 + 0.4 * progress);
            elseif generation <= obj.SteadyStateGenerations
                % Phase 3: Steady-state
                progress = (generation - obj.WarmupGenerations) / ...
                          (obj.SteadyStateGenerations - obj.WarmupGenerations);
                ratio = obj.MaxDMRatio * (0.5 + 0.5 * progress);
            else
                % Phase 4: Adaptive - maintain high ratio
                ratio = obj.MaxDMRatio;
            end
        end
        
        %% Quality-based modulation factor
        function factor = getQualityFactor(obj, metrics)
            if ~isfield(metrics, 'dmSurvivalRate') || metrics.dmSurvivalRate == 0
                factor = 1.0;
                return;
            end
            
            survivalRate = metrics.dmSurvivalRate;
            
            % If DM offspring are performing well, increase ratio
            % If performing poorly, decrease ratio
            targetRate = 0.3;  % Target survival rate
            factor = min(1.5, max(0.5, survivalRate / targetRate));
        end
        
        %% Stagnation-based reduction factor
        function factor = getStagnationFactor(obj, metrics)
            if ~isfield(metrics, 'stagnationGenerations')
                factor = 1.0;
                return;
            end
            
            stagnationGens = metrics.stagnationGenerations;
            
            if stagnationGens > 10
                % Severe stagnation - reduce DM contribution significantly
                factor = 0.5;
            elseif stagnationGens > 5
                % Moderate stagnation - slight reduction
                factor = 0.7;
            else
                % No stagnation - normal operation
                factor = 1.0;
            end
        end
        
        %% Exploration factor
        function factor = getExplorationFactor(obj, metrics)
            % Increase diversity when archive is not growing
            if isfield(metrics, 'archiveSize') && metrics.generation > 20
                % Check if archive has been stagnant
                if length(obj.QualityHistory) >= 10
                    recentQuality = mean(obj.QualityHistory(end-4:end));
                    oldQuality = mean(obj.QualityHistory(1:5));
                    
                    if recentQuality < oldQuality * 0.8
                        % Quality declining - increase exploration
                        factor = 1.2;
                    else
                        factor = 1.0;
                    end
                else
                    factor = 1.0;
                end
            else
                factor = 1.0;
            end
        end
        
        %% Sigmoid helper function
        function y = sigmoid(obj, x)
            y = 1 / (1 + exp(-x));
        end
    end
end
