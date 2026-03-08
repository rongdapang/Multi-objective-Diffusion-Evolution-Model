classdef AdaptiveScheduler < handle
% AdaptiveScheduler - Adaptive scheduler for GA/DM ratio adjustment
%
% This class dynamically adjusts the ratio between GA and Diffusion Model
% offspring generation based on optimization progress and DM performance.
%
% Properties:
%   BaseDMRatio - Base ratio of DM offspring
%   MinDMRatio - Minimum DM ratio
%   MaxDMRatio - Maximum DM ratio
%   AdaptationRate - Rate of adaptation

    properties (SetAccess = private)
        BaseDMRatio     % Base ratio of DM offspring
        MinDMRatio      % Minimum DM ratio
        MaxDMRatio      % Maximum DM ratio
        History         % History of DM performance
        WindowSize      % Window size for moving average
    end
    
    methods
        %% Constructor
        function obj = AdaptiveScheduler(baseRatio, minRatio, maxRatio)
            obj.BaseDMRatio = baseRatio;
            obj.MinDMRatio = minRatio;
            obj.MaxDMRatio = maxRatio;
            obj.History = [];
            obj.WindowSize = 10;
        end
        
        %% Get current DM ratio
        function dmRatio = getDMRatio(obj, generation, metrics)
            % Start with base ratio
            dmRatio = obj.BaseDMRatio;
            
            % Adjust based on generation (early generations favor GA)
            if generation < 20
                dmRatio = dmRatio * (generation / 20);
            end
            
            % Adjust based on DM survival rate
            if isfield(metrics, 'dmSurvivalRate')
                survivalRate = metrics.dmSurvivalRate;
                
                % If DM is performing well, increase its ratio
                if survivalRate > 0.4
                    dmRatio = dmRatio * 1.1;
                elseif survivalRate < 0.2
                    dmRatio = dmRatio * 0.9;
                end
                
                % Store in history
                obj.History = [obj.History, survivalRate];
                if length(obj.History) > obj.WindowSize
                    obj.History = obj.History(end-obj.WindowSize+1:end);
                end
            end
            
            % Adjust based on stagnation
            if isfield(metrics, 'stagnationGenerations') && metrics.stagnationGenerations > 5
                % If stagnating, reduce DM ratio to explore more with GA
                dmRatio = dmRatio * 0.8;
            end
            
            % Clamp to valid range
            dmRatio = max(obj.MinDMRatio, min(obj.MaxDMRatio, dmRatio));
            dmRatio = max(0, min(1, dmRatio));
        end
        
        %% Get performance statistics
        function stats = getStats(obj)
            if isempty(obj.History)
                stats.mean = 0;
                stats.std = 0;
                stats.trend = 0;
            else
                stats.mean = mean(obj.History);
                stats.std = std(obj.History);
                if length(obj.History) >= 3
                    stats.trend = mean(diff(obj.History(end-2:end)));
                else
                    stats.trend = 0;
                end
            end
        end
        
        %% Reset history
        function reset(obj)
            obj.History = [];
        end
    end
end
