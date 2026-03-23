classdef SolutionArchive < handle
% SolutionArchive - 优化版本（批量添加 + 快速去重 + 智能剪枝）

    properties(Access = private)
        MaxSize
        M
        Solutions
        HVValues
        RefPoint
        PendingSolutions      % 新增：待处理解缓存
        BatchThreshold        % 新增：批量处理阈值
        LastPruneGen          % 新增：上次剪枝的代数
        HVThreshold           % 新增：hypervolume 策略阈值
    end

    methods
        function obj = SolutionArchive(max_size, M)
            obj.MaxSize = max_size;
            obj.M = M;
            obj.Solutions = [];
            obj.HVValues = [];
            obj.RefPoint = ones(1, M) * 1.1;
            obj.PendingSolutions = [];
            obj.BatchThreshold = max(50, floor(max_size * 0.05));
            obj.LastPruneGen = 0;
            obj.HVThreshold = 400;
        end

        function add(obj, NewSolutions)
            if isempty(NewSolutions)
                return;
            end
            currentSize = length(obj.Solutions);
            if currentSize + length(NewSolutions) <= obj.MaxSize * 0.9
                if isempty(obj.Solutions)
                    obj.Solutions = obj.removeDuplicatesFast(NewSolutions);
                else
                    Combined = [obj.Solutions, NewSolutions];
                    obj.Solutions = obj.removeDuplicatesFast(Combined);
                end
                if ~isempty(obj.Solutions)
                    obj.RefPoint = max(obj.Solutions.objs, [], 1) * 1.1;
                    obj.RefPoint(obj.RefPoint == 0) = 1;
                end
                disp(['      Archive.add: Added ', num2str(length(NewSolutions)), ' solutions, now archive size = ', num2str(length(obj.Solutions)), ' (FAST PATH)']);
                return;
            end
            
            if isempty(obj.Solutions)
                Combined = NewSolutions;
            else
                Combined = [obj.Solutions, NewSolutions];
            end
            Combined = obj.removeDuplicatesFast(Combined);
            if length(Combined) > obj.MaxSize
                Combined = obj.selectByHybridStrategy(Combined);
            end
            obj.Solutions = Combined;
            disp(['      Archive.add: Added ', num2str(length(NewSolutions)), ' solutions, now archive size = ', num2str(length(obj.Solutions))]);
            if ~isempty(obj.Solutions)
                obj.RefPoint = max(obj.Solutions.objs, [], 1) * 1.1;
                obj.RefPoint(obj.RefPoint == 0) = 1;
            end
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
                    obj.PendingSolutions = obj.selectDiverseSolutions(obj.PendingSolutions, obj.BatchThreshold * 2);
                    disp(['      Archive.addBatch: Trimmed pending queue to ', num2str(length(obj.PendingSolutions)), ' (diversity-based)']);
                else
                    disp(['      Archive.addBatch: Cached ', num2str(length(NewSolutions)), ' solutions, pending = ', num2str(length(obj.PendingSolutions))]);
                end
            end
        end

        function processPending(obj, currentGen)
            if isempty(obj.PendingSolutions)
                return;
            end
            if isempty(obj.Solutions)
                Combined = obj.PendingSolutions;
            else
                Combined = [obj.Solutions, obj.PendingSolutions];
            end
            obj.PendingSolutions = [];
            Combined = obj.removeDuplicatesFast(Combined);
            if length(Combined) > obj.MaxSize
                Combined = obj.selectByHybridStrategy(Combined);
            end
            obj.Solutions = Combined;
            obj.LastPruneGen = currentGen;
            disp(['      Archive.processPending: Processed batch, archive size = ', num2str(length(obj.Solutions))]);
            if ~isempty(obj.Solutions)
                obj.RefPoint = max(obj.Solutions.objs, [], 1) * 1.1;
                obj.RefPoint(obj.RefPoint == 0) = 1;
            end
        end

        function Unique = removeDuplicatesFast(~, Solutions)
            if isempty(Solutions)
                Unique = Solutions;
                return;
            end
            n = length(Solutions);
            if n <= 1
                Unique = Solutions;
                return;
            end
            try
                decs_raw = {Solutions.dec};
                decs = cell2mat(decs_raw');
            catch ME
                Unique = Solutions;
                disp(['      [removeDuplicatesFast] Warning: Could not extract decs: ' ME.message ', returning all ', num2str(n), ' solutions']);
                return;
            end
            tolerance = 1e-4;
            decs_rounded = round(decs / tolerance) * tolerance;
            [~, unique_idx] = unique(decs_rounded, 'rows', 'stable');
            Unique = Solutions(unique_idx);
            if length(Unique) < n
                disp(['      [removeDuplicatesFast] Removed ', num2str(n - length(Unique)), ' duplicates, kept ', num2str(length(Unique))]);
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

        function Selected = selectByNDSort(obj, Solutions)
            n = length(Solutions);
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
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
                    remaining_slots = obj.MaxSize - length(Selected);
                    if length(front_solutions) > remaining_slots
                        front_solutions = obj.selectDiverseSolutions(front_solutions, remaining_slots);
                    end
                end
                Selected = [Selected, front_solutions];
                current_front = current_front + 1;
            end
        end

        function Selected = selectDiverseSolutions(obj, Solutions, n_select)
            n = length(Solutions);
            if n <= n_select
                Selected = Solutions;
                return;
            end
            objs = Solutions.objs;
            min_obj = min(objs, [], 1);
            max_obj = max(objs, [], 1);
            range_obj = max_obj - min_obj;
            range_obj(range_obj == 0) = 1;
            norm_objs = (objs - min_obj) ./ range_obj;
            Selected = false(1, n);
            for m = 1:obj.M
                [~, min_idx] = min(objs(:, m));
                if ~Selected(min_idx)
                    Selected(min_idx) = true;
                end
                if sum(Selected) >= n_select
                    break;
                end
                [~, max_idx] = max(objs(:, m));
                if ~Selected(max_idx)
                    Selected(max_idx) = true;
                end
                if sum(Selected) >= n_select
                    break;
                end
            end
            while sum(Selected) < n_select
                selected_idx = find(Selected);
                not_selected_idx = find(~Selected);
                if isempty(not_selected_idx)
                    break;
                end
                min_distances = inf(1, length(not_selected_idx));
                for i = 1:length(not_selected_idx)
                    idx = not_selected_idx(i);
                    distances = sqrt(sum((norm_objs(selected_idx, :) - norm_objs(idx, :)).^2, 2));
                    min_distances(i) = min(distances);
                end
                [~, best_idx] = max(min_distances);
                Selected(not_selected_idx(best_idx)) = true;
            end
            Selected = Solutions(Selected);
        end

        function Selected = selectByHybridStrategy(obj, Solutions)
            n = length(Solutions);
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            if n < obj.HVThreshold
                disp(['      [HYBRID] Using Hypervolume selection (n=', num2str(n), ' < threshold=', num2str(obj.HVThreshold), ')']);
                Selected = obj.selectByHypervolume(Solutions);
            else
                disp(['      [HYBRID] Using NDSort selection (n=', num2str(n), ' >= threshold=', num2str(obj.HVThreshold), ')']);
                Selected = obj.selectByNDSort(Solutions);
            end
        end

        function Selected = selectByHypervolume(obj, Solutions)
            n = length(Solutions);
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            hv_full = obj.calculateHypervolume(Solutions);
            hv_contrib = zeros(1, n);
            for i = 1:n
                without_i = setdiff(1:n, i);
                if isempty(without_i)
                    hv_contrib(i) = inf;
                else
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
            n_samples = 2000;
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
            if isempty(obj.Solutions)
                Data = [];
                return;
            end
            total_count = length(obj.Solutions);
            if total_count < 10
                Data = obj.Solutions;
                return;
            end
            [FrontNo, ~] = NDSort(obj.Solutions.objs, obj.Solutions.cons, total_count);
            non_dominated = obj.Solutions(FrontNo == 1);
            non_dominated_count = length(non_dominated);
            if non_dominated_count < 10
                Data = obj.Solutions;
                disp(['      [TrainingData] Using all solutions: ', num2str(total_count), ' (non-dominated only ', num2str(non_dominated_count), ', too few)']);
            else
                Data = non_dominated;
                ratio = 100 * non_dominated_count / total_count;
                disp(['      [TrainingData] Filtered: ', num2str(total_count), ' total → ', num2str(non_dominated_count), ' non-dominated (', sprintf('%.1f', ratio), '%)']);
            end
        end

        function size = getSize(obj)
            size = length(obj.Solutions);
        end

        function clear(obj)
            obj.Solutions = [];
            obj.HVValues = [];
            obj.PendingSolutions = [];
            obj.LastPruneGen = 0;
        end

        function setMaxSize(obj, new_max_size)
            obj.MaxSize = new_max_size;
            obj.BatchThreshold = max(50, floor(new_max_size * 0.05));
            if length(obj.Solutions) > new_max_size
                obj.Solutions = obj.selectByHybridStrategy(obj.Solutions);
            end
        end

        function count = getPendingCount(obj)
            count = length(obj.PendingSolutions);
        end
    end
end
