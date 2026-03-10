classdef SolutionArchive < handle
% SolutionArchive - Elite solution archive with hypervolume contribution

    properties(Access = private)
        MaxSize
        M
        Solutions
        HVValues
        RefPoint
    end

    methods
        function obj = SolutionArchive(max_size, M)
            obj.MaxSize = max_size;
            obj.M = M;
            obj.Solutions = [];
            obj.HVValues = [];
            obj.RefPoint = ones(1, M) * 1.1;
        end
        
        function add(obj, NewSolutions)
            if isempty(NewSolutions)
                return;
            end
            
            if isempty(obj.Solutions)
                Combined = NewSolutions;
            else
                Combined = [obj.Solutions, NewSolutions];
            end
            
            Combined = obj.removeDuplicates(Combined);
            
            if length(Combined) > obj.MaxSize
                Combined = obj.selectByHypervolume(Combined);
            end
            
            obj.Solutions = Combined;
            
            if ~isempty(obj.Solutions)
                obj.RefPoint = max(obj.Solutions.objs, [], 1) * 1.1;
                obj.RefPoint(obj.RefPoint == 0) = 1;
            end
        end
        
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
        
        function Selected = selectByHypervolume(obj, Solutions)
            n = length(Solutions);
            
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            
            hv_contrib = zeros(1, n);
            
            for i = 1:n
                without_i = setdiff(1:n, i);
                if isempty(without_i)
                    hv_contrib(i) = inf;
                else
                    hv_full = obj.calculateHypervolume(Solutions);
                    hv_without = obj.calculateHypervolume(Solutions(without_i));
                    hv_contrib(i) = hv_full - hv_without;
                end
            end
            
            [~, idx] = sort(hv_contrib, 'descend');
            Selected = Solutions(idx(1:obj.MaxSize));
        end
        
        function hv = calculateHypervolume(obj, Solutions)
            if isempty(Solutions)
                hv = 0;
                return;
            end
            
            objs = Solutions.objs;
            
            if obj.M == 2
                hv = obj.calculateHV2D(objs);
            else
                hv = obj.calculateHVMonteCarlo(objs);
            end
        end
        
        function hv = calculateHV2D(~, objs)
            [sorted, idx] = sort(objs(:, 1));
            sorted_objs = objs(idx, :);
            
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
        
        function hv = calculateHVMonteCarlo(obj, objs)
            n_samples = 10000;
            samples = rand(n_samples, obj.M) .* obj.RefPoint;
            
            dominated = 0;
            for i = 1:n_samples
                for j = 1:size(objs, 1)
                    if all(samples(i, :) <= objs(j, :))
                        dominated = dominated + 1;
                        break;
                    end
                end
            end
            
            volume = prod(obj.RefPoint);
            hv = volume * dominated / n_samples;
        end
        
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
                try
                    [~, C] = kmeans(objs, n_targets, 'MaxIter', 100);
                    Targets = C;
                catch
                    idx = randperm(n_available, n_targets);
                    Targets = objs(idx, :);
                end
            end
        end
        
        function Objs = getObjectives(obj)
            if isempty(obj.Solutions)
                Objs = [];
            else
                Objs = obj.Solutions.objs;
            end
        end
        
        function Data = getTrainingData(obj)
            Data = obj.Solutions;
        end
        
        function size = getSize(obj)
            size = length(obj.Solutions);
        end
        
        function clear(obj)
            obj.Solutions = [];
            obj.HVValues = [];
        end
    end
end
