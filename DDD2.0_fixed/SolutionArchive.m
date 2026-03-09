classdef SolutionArchive < handle
% SolutionArchive - Elite solution archive with hypervolume contribution
%
% This class manages an archive of elite solutions using hypervolume
% contribution for quality assessment and bounded size maintenance.

    properties(Access = private)
        MaxSize     % Maximum archive size
        M           % Number of objectives
        Solutions   % Stored solutions
        HVValues    % Hypervolume contributions
        RefPoint    % Reference point for HV calculation
    end

    methods
        %% Constructor
        function obj = SolutionArchive(max_size, M)
            obj.MaxSize = max_size;
            obj.M = M;
            obj.Solutions = [];
            obj.HVValues = [];
            obj.RefPoint = ones(1, M) * 1.1;  % Default reference point
        end
        
        %% Add solutions to archive
        function add(obj, NewSolutions)
            if isempty(NewSolutions)
                return;
            end
            
            % Combine with existing solutions
            if isempty(obj.Solutions)
                Combined = NewSolutions;
            else
                Combined = [obj.Solutions, NewSolutions];
            end
            
            % Remove duplicates based on decision variables
            Combined = obj.removeDuplicates(Combined);
            
            % If still too many, select based on hypervolume contribution
            if length(Combined) > obj.MaxSize
                Combined = obj.selectByHypervolume(Combined);
            end
            
            obj.Solutions = Combined;
            
            % Update reference point
            if ~isempty(obj.Solutions)
                obj.RefPoint = max(obj.Solutions.objs, [], 1) * 1.1;
                obj.RefPoint(obj.RefPoint == 0) = 1;
            end
        end
        
        %% Remove duplicate solutions
        function Unique = removeDuplicates(~, Solutions)
            if isempty(Solutions)
                Unique = Solutions;
                return;
            end
            
            n = length(Solutions);
            keep = true(1, n);
            
            for i = 1:n-1
                if ~keep(i)
                    continue;
                end
                for j = i+1:n
                    if keep(j) && isequal(Solutions(i).dec, Solutions(j).dec)
                        keep(j) = false;
                    end
                end
            end
            
            Unique = Solutions(keep);
        end
        
        %% Select solutions by hypervolume contribution
        function Selected = selectByHypervolume(obj, Solutions)
            n = length(Solutions);
            
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            
            % Calculate hypervolume contribution for each solution
            hv_contrib = zeros(1, n);
            
            for i = 1:n
                % Calculate HV without this solution
                without_i = setdiff(1:n, i);
                if isempty(without_i)
                    hv_contrib(i) = inf;
                else
                    hv_full = obj.calculateHypervolume(Solutions);
                    hv_without = obj.calculateHypervolume(Solutions(without_i));
                    hv_contrib(i) = hv_full - hv_without;
                end
            end
            
            % Keep solutions with highest contributions
            [~, idx] = sort(hv_contrib, 'descend');
            Selected = Solutions(idx(1:obj.MaxSize));
        end
        
        %% Calculate hypervolume (simplified for 2-3 objectives)
        function hv = calculateHypervolume(obj, Solutions)
            if isempty(Solutions)
                hv = 0;
                return;
            end
            
            objs = Solutions.objs;
            
            % For 2 objectives, use exact calculation
            if obj.M == 2
                hv = obj.calculateHV2D(objs);
            else
                % For higher dimensions, use Monte Carlo approximation
                hv = obj.calculateHVMonteCarlo(objs);
            end
        end
        
        %% 2D hypervolume calculation
        function hv = calculateHV2D(~, objs)
            % Sort by first objective
            [sorted, idx] = sort(objs(:, 1));
            sorted_objs = objs(idx, :);
            
            % Calculate hypervolume
            hv = 0;
            ref = [1.1, 1.1];
            
            for i = 1:size(sorted_objs, 1)
                if i == 1
                    width = ref(1) - sorted_objs(i, 1);
                else
                    width = sorted_objs(i-1, 1) - sorted_objs(i, 1);
                end
                height = ref(2) - sorted_objs(i, 2);
                hv = hv + width * height;
            end
        end
        
        %% Monte Carlo hypervolume approximation
        function hv = calculateHVMonteCarlo(obj, objs)
            n_samples = 10000;
            
            % Sample points in the objective space
            samples = rand(n_samples, obj.M) .* obj.RefPoint;
            
            % Count dominated points
            dominated = 0;
            for i = 1:n_samples
                for j = 1:size(objs, 1)
                    if all(samples(i, :) <= objs(j, :))
                        dominated = dominated + 1;
                        break;
                    end
                end
            end
            
            % Hypervolume = volume * dominated ratio
            volume = prod(obj.RefPoint);
            hv = volume * dominated / n_samples;
        end
        
        %% Get target objectives for sampling
        function Targets = getTargetObjectives(obj, n_targets)
            if isempty(obj.Solutions)
                Targets = [];
                return;
            end
            
            objs = obj.Solutions.objs;
            n_available = size(objs, 1);
            
            if n_available <= n_targets
                Targets = objs;
            else
                % Select diverse targets
                try
                    % Use k-means clustering
                    [~, C] = kmeans(objs, n_targets, 'MaxIter', 100);
                    Targets = C;
                catch
                    % Fallback to random selection
                    idx = randperm(n_available, n_targets);
                    Targets = objs(idx, :);
                end
            end
        end
        
        %% Get all objectives in archive
        function Objs = getObjectives(obj)
            if isempty(obj.Solutions)
                Objs = [];
            else
                Objs = obj.Solutions.objs;
            end
        end
        
        %% Get training data
        function Data = getTrainingData(obj)
            Data = obj.Solutions;
        end
        
        %% Get archive size
        function size = getSize(obj)
            size = length(obj.Solutions);
        end
        
        %% Clear archive
        function clear(obj)
            obj.Solutions = [];
            obj.HVValues = [];
        end
    end
end
