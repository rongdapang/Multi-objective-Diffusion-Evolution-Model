classdef AdaptiveScheduler < handle
% AdaptiveScheduler - Adaptive scheduler for GA/DM offspring ratio

    properties(Access = private)
        BaseDMRatio
        MinDMRatio
        MaxDMRatio
        WindowSize
        DMHistory
        GAHistory
        StagnationCount
        LastBest
    end

    methods
        function obj = AdaptiveScheduler(base_ratio, min_ratio, max_ratio)
            obj.BaseDMRatio = base_ratio;
            obj.MinDMRatio = min_ratio;
            obj.MaxDMRatio = max_ratio;
            obj.WindowSize = 10;
            obj.DMHistory = [];
            obj.GAHistory = [];
            obj.StagnationCount = 0;
            obj.LastBest = inf;
        end

        function count = getDMOffspringCount(obj, total_offspring, dm_stats)
            current_ratio = obj.calculateDMRatio(dm_stats);
            count = round(total_offspring * current_ratio);
            if count >= total_offspring
                count = total_offspring - 1;
            end
            count = max(0, count);
        end

        function ratio = calculateDMRatio(obj, dm_stats)
            ratio = obj.BaseDMRatio;
            if ~isempty(dm_stats.history) && length(dm_stats.history) >= 3
                recent_performance = mean(dm_stats.history(end-2:end));
                if recent_performance > 0.3
                    ratio = ratio * 1.2;
                elseif recent_performance < 0.1
                    ratio = ratio * 0.8;
                end
            end
            if obj.StagnationCount > 5
                ratio = ratio * 0.9;
            end
            ratio = max(obj.MinDMRatio, min(obj.MaxDMRatio, ratio));
        end

        function updatePerformance(obj, population, ~)
            current_best = min(population.objs(:, 1));
            if current_best >= obj.LastBest
                obj.StagnationCount = obj.StagnationCount + 1;
            else
                obj.StagnationCount = 0;
            end
            obj.LastBest = min(obj.LastBest, current_best);
        end

        function resetStagnation(obj)
            obj.StagnationCount = 0;
        end

        function count = getStagnationCount(obj)
            count = obj.StagnationCount;
        end

        function setBaseRatio(obj, ratio)
            obj.BaseDMRatio = ratio;
        end

        function stats = getStats(obj)
            stats = struct('base_ratio', obj.BaseDMRatio, ...
                          'stagnation_count', obj.StagnationCount, ...
                          'dm_history_length', length(obj.DMHistory));
        end
    end
end
