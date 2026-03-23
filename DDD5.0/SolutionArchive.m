classdef SolutionArchive < handle
    properties(Access = private)
        MaxSize
        M
        Solutions
        PendingSolutions
        BatchThreshold
        LastPruneGen
        HVThreshold
        DedupTolerance
        ObjDedupTolerance
    end

    methods
        function obj = SolutionArchive(max_size, M)
            obj.MaxSize = max_size;
            obj.M = M;
            obj.Solutions = [];
            obj.PendingSolutions = [];
            obj.BatchThreshold = max(50, floor(max_size * 0.05));
            obj.LastPruneGen = 0;
            obj.HVThreshold = obj.calculateAdaptiveHVThreshold(max_size, M);
            obj.DedupTolerance = 1e-6;
            obj.ObjDedupTolerance = 1e-4;
        end

        function threshold = calculateAdaptiveHVThreshold(~, max_size, M)
            if M > 3
                threshold = 0;
            elseif max_size > 1000
                threshold = 0;
            else
                threshold = 500 * (3 / M) * (1000 / max_size);
                threshold = max(100, min(1000, threshold));
            end
        end

        function add(obj, NewSolutions)
            if isempty(NewSolutions)
                return;
            end
            Combined = [obj.Solutions, NewSolutions];
            Combined = obj.removeDuplicates(Combined);
            if length(Combined) > obj.MaxSize
                Combined = obj.selectByHybridStrategy(Combined);
            end
            obj.Solutions = Combined;
        end

        function addBatch(obj, NewSolutions, currentGen)
            if isempty(NewSolutions)
                return;
            end
            obj.PendingSolutions = [obj.PendingSolutions, NewSolutions];
            totalPending = length(obj.Solutions) + length(obj.PendingSolutions);
            shouldProcess = false;
            if length(obj.PendingSolutions) >= obj.BatchThreshold
                shouldProcess = true;
            end
            if totalPending > obj.MaxSize * 1.2
                shouldProcess = true;
            end
            if ~isempty(currentGen) && mod(currentGen, 5) == 0
                shouldProcess = true;
            end
            
            if shouldProcess
                obj.processPending(currentGen);
            else
                if length(obj.PendingSolutions) > obj.BatchThreshold * 2
                    obj.PendingSolutions = obj.PendingSolutions(1:obj.BatchThreshold*2);
                end
            end
        end

        function processPending(obj, currentGen)
            if isempty(obj.PendingSolutions)
                return;
            end
            Combined = [obj.Solutions, obj.PendingSolutions];
            obj.PendingSolutions = [];
            Combined = obj.removeDuplicates(Combined);
            if length(Combined) > obj.MaxSize
                Combined = obj.selectByHybridStrategy(Combined);
            end
            obj.Solutions = Combined;
            obj.LastPruneGen = currentGen;
        end

        function Unique = removeDuplicates(obj, Solutions)
            if isempty(Solutions) || length(Solutions) <= 1
                Unique = Solutions;
                return;
            end
            n = length(Solutions);
            keep = true(1, n);
            
            decs = zeros(n, length(Solutions(1).dec));
            objs = zeros(n, length(Solutions(1).objs));
            
            for i = 1:n
                decs(i,:) = Solutions(i).dec;
                objs(i,:) = Solutions(i).objs;
            end
            
            for i = 1:n-1
                if ~keep(i), continue; end
                for j = i+1:n
                    if keep(j) && norm(decs(i,:) - decs(j,:)) < obj.DedupTolerance
                        if norm(objs(i,:) - objs(j,:)) < obj.ObjDedupTolerance
                            keep(j) = false;
                        end
                    end
                end
            end
            Unique = Solutions(keep);
        end

        function Selected = selectByHybridStrategy(obj, Solutions)
            n = length(Solutions);
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            
            current_threshold = obj.calculateAdaptiveHVThreshold(n, obj.M);
            
            if n < current_threshold && obj.M <= 3
                Selected = obj.selectByHypervolume(Solutions);
            else
                Selected = obj.selectByNDSort(Solutions);
            end
        end

        function Selected = selectByNDSort(obj, Solutions)
            n = length(Solutions);
            [FrontNo, ~] = NDSort(Solutions.objs, Solutions.cons, n);
            Selected = [];
            current_front = 1;
            while length(Selected) < obj.MaxSize && current_front <= max(FrontNo)
                front_idx = find(FrontNo == current_front);
                front_solutions = Solutions(front_idx);
                if isempty(front_solutions)
                    current_front = current_front + 1;
                    continue;
                end
                if length(Selected) + length(front_solutions) > obj.MaxSize
                    remaining = obj.MaxSize - length(Selected);
                    front_solutions = obj.selectDiverse(front_solutions, remaining);
                end
                Selected = [Selected, front_solutions];
                current_front = current_front + 1;
            end
        end

        function Selected = selectDiverse(obj, Solutions, n_select)
            n = length(Solutions);
            if n <= n_select
                Selected = Solutions;
                return;
            end
            objs = Solutions.objs;
            
            min_obj = min(objs, [], 1);
            max_obj = max(objs, [], 1);
            range = max_obj - min_obj;
            range(range == 0) = 1;
            norm_objs = (objs - min_obj) ./ range;
            
            selected = false(1, n);
            
            for m = 1:obj.M
                [~, min_idx] = min(objs(:, m));
                selected(min_idx) = true;
                [~, max_idx] = max(objs(:, m));
                selected(max_idx) = true;
            end
            
            while sum(selected) < n_select
                not_selected = find(~selected);
                if isempty(not_selected), break; end
                
                distances = zeros(1, length(not_selected));
                for i = 1:length(not_selected)
                    idx = not_selected(i);
                    if sum(selected) > 0
                        dists = sqrt(sum((norm_objs(selected, :) - norm_objs(idx, :)).^2, 2));
                        distances(i) = min(dists);
                    else
                        distances(i) = inf;
                    end
                end
                [~, best] = max(distances);
                selected(not_selected(best)) = true;
            end
            Selected = Solutions(selected);
        end

        function Selected = selectByHypervolume(obj, Solutions)
            n = length(Solutions);
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            
            ref = max(Solutions.objs, [], 1) * 1.1;
            hv_full = obj.calcHypervolume(Solutions, ref);
            contributions = zeros(1, n);
            
            for i = 1:n
                without = Solutions;
                without(i) = [];
                hv_without = obj.calcHypervolume(without, ref);
                contributions(i) = hv_full - hv_without;
            end
            
            [~, idx] = sort(contributions, 'descend');
            Selected = Solutions(idx(1:obj.MaxSize));
        end

        function hv = calcHypervolume(obj, Solutions, ref)
            if isempty(Solutions)
                hv = 0;
                return;
            end
            objs = Solutions.objs;
            if obj.M == 2
                sorted = sortrows(objs, 1);
                hv = 0;
                for i = 1:size(sorted,1)
                    if i == 1
                        width = ref(1) - sorted(i,1);
                    else
                        width = sorted(i-1,1) - sorted(i,1);
                    end
                    height = ref(2) - sorted(i,2);
                    hv = hv + width * height;
                end
            else
                n_samples = 2000;
                samples = rand(n_samples, obj.M) .* ref;
                dominated = 0;
                for i = 1:n_samples
                    if any(all(samples(i,:) <= objs, 2))
                        dominated = dominated + 1;
                    end
                end
                hv = prod(ref) * dominated / n_samples;
            end
        end

        function Targets = getTargetObjectives(obj, n_targets)
            if isempty(obj.Solutions)
                Targets = [];
                return;
            end
            objs = obj.Solutions.objs;
            n = size(objs,1);
            if n <= n_targets
                Targets = objs;
                return;
            end
            try
                [~, C] = kmeans(objs, n_targets, 'MaxIter', 100);
                Targets = C;
            catch
                idx = randperm(n, n_targets);
                Targets = objs(idx, :);
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
            if isempty(obj.Solutions)
                Data = [];
                return;
            end
            [FrontNo, ~] = NDSort(obj.Solutions.objs, obj.Solutions.cons, length(obj.Solutions));
            Data = obj.Solutions(FrontNo == 1);
            if length(Data) < 10
                Data = obj.Solutions;
            end
        end

        function size = getSize(obj)
            size = length(obj.Solutions);
        end

        function clear(obj)
            obj.Solutions = [];
            obj.PendingSolutions = [];
        end

        function setMaxSize(obj, new_max)
            obj.MaxSize = new_max;
            obj.BatchThreshold = max(50, floor(new_max * 0.05));
            obj.HVThreshold = obj.calculateAdaptiveHVThreshold(new_max, obj.M);
            if length(obj.Solutions) > new_max
                obj.Solutions = obj.selectByHybridStrategy(obj.Solutions);
            end
        end

        function count = getPendingCount(obj)
            count = length(obj.PendingSolutions);
        end
    end
end