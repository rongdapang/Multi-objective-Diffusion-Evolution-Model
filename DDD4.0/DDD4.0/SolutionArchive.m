classdef SolutionArchive < handle
% SolutionArchive - 优化版本（批量添加 + 快速去重 + 智能剪枝）
% 优化日期: 2026-03-12
% 
% 优化内容:
% 1. 批量添加机制：累积解到缓冲区，达到阈值时才剪枝
% 2. 快速去重：使用排序替代 O(n^2) 双重循环
% 3. 智能剪枝策略：使用非支配排序+拥挤距离替代昂贵的 hypervolume 计算
% 4. 预期性能：Archive Add 耗时从 60s+ 降至 <1s

    properties(Access = private)
        MaxSize
        M
        Solutions
        HVValues
        RefPoint
        PendingSolutions      % 新增：待处理解缓存
        BatchThreshold        % 新增：批量处理阈值
        LastPruneGen          % 新增：上次剪枝的代数
        HVThreshold           % 新增：hypervolume策略阈值
    end

    methods
        function obj = SolutionArchive(max_size, M)
            obj.MaxSize = max_size;
            obj.M = M;
            obj.Solutions = [];
            obj.HVValues = [];
            obj.RefPoint = ones(1, M) * 1.1;
            obj.PendingSolutions = [];    % 初始化待处理缓存
            obj.BatchThreshold = max(50, floor(max_size * 0.05));  % 批量阈值：5%容量或至少50
            obj.LastPruneGen = 0;
            obj.HVThreshold = 400;        % 混合策略阈值：小于此值用hypervolume，大于等于用NDSort
        end
        
        function add(obj, NewSolutions)
            if isempty(NewSolutions)
                return;
            end
            
            % 快速路径：如果 Archive 远未满，直接添加不剪枝
            currentSize = length(obj.Solutions);
            if currentSize + length(NewSolutions) <= obj.MaxSize * 0.9
                % 直接合并，只去重，不剪枝
                if isempty(obj.Solutions)
                    obj.Solutions = obj.removeDuplicatesFast(NewSolutions);
                else
                    Combined = [obj.Solutions, NewSolutions];
                    obj.Solutions = obj.removeDuplicatesFast(Combined);
                end
                
                % 更新参考点
                if ~isempty(obj.Solutions)
                    obj.RefPoint = max(obj.Solutions.objs, [], 1) * 1.1;
                    obj.RefPoint(obj.RefPoint == 0) = 1;
                end
                
                disp(['      Archive.add: Added ', num2str(length(NewSolutions)), ' solutions, now archive size = ', num2str(length(obj.Solutions)), ' (FAST PATH)']);
                return;
            end
            
            % 标准路径：需要剪枝
            if isempty(obj.Solutions)
                Combined = NewSolutions;
            else
                Combined = [obj.Solutions, NewSolutions];
            end
            
            % 使用快速去重
            Combined = obj.removeDuplicatesFast(Combined);
            
            % 混合策略剪枝：根据解集大小选择hypervolume或NDSort
            if length(Combined) > obj.MaxSize
                Combined = obj.selectByHybridStrategy(Combined);
            end
            
            obj.Solutions = Combined;
            
            % 显示调试信息
            disp(['      Archive.add: Added ', num2str(length(NewSolutions)), ' solutions, now archive size = ', num2str(length(obj.Solutions))]);
            
            if ~isempty(obj.Solutions)
                obj.RefPoint = max(obj.Solutions.objs, [], 1) * 1.1;
                obj.RefPoint(obj.RefPoint == 0) = 1;
            end
        end
        
        % 新增：批量添加方法（优化版：定期维护多样性）
        function addBatch(obj, NewSolutions, currentGen)
            if isempty(NewSolutions)
                return;
            end
            
            % 添加到待处理队列
            obj.PendingSolutions = [obj.PendingSolutions, NewSolutions];
            
            % 检查是否需要处理
            totalPending = length(obj.Solutions) + length(obj.PendingSolutions);
            shouldProcess = false;
            
            % 条件1：待处理数量超过阈值
            if length(obj.PendingSolutions) >= obj.BatchThreshold
                shouldProcess = true;
            end
            
            % 条件2：总数量超过容量 120%
            if totalPending > obj.MaxSize * 1.2
                shouldProcess = true;
            end
            
            % 条件3：每隔5代强制处理（原来是10代，提高多样性维护频率）
            if ~isempty(currentGen) && mod(currentGen, 5) == 0
                shouldProcess = true;
            end
            
            if shouldProcess
                obj.processPending(currentGen);
            else
                % 优化：即使不处理，也限制待处理队列大小，防止过度累积
                if length(obj.PendingSolutions) > obj.BatchThreshold * 2
                    % 使用多样性选择截断待处理队列
                    obj.PendingSolutions = obj.selectDiverseSolutions(obj.PendingSolutions, obj.BatchThreshold * 2);
                    disp(['      Archive.addBatch: Trimmed pending queue to ', num2str(length(obj.PendingSolutions)), ' (diversity-based)']);
                else
                    disp(['      Archive.addBatch: Cached ', num2str(length(NewSolutions)), ' solutions, pending = ', num2str(length(obj.PendingSolutions))]);
                end
            end
        end
        
        % 新增：处理待处理队列
        function processPending(obj, currentGen)
            if isempty(obj.PendingSolutions)
                return;
            end
            
            % 合并所有解
            if isempty(obj.Solutions)
                Combined = obj.PendingSolutions;
            else
                Combined = [obj.Solutions, obj.PendingSolutions];
            end
            
            % 清空待处理队列
            obj.PendingSolutions = [];
            
            % 去重
            Combined = obj.removeDuplicatesFast(Combined);
            
            % 混合策略剪枝
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
        
        % 优化：快速去重（使用排序，O(n log n)）
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
            
            % 提取决策变量并转换为矩阵
            try
                % Solutions.dec 可能返回矩阵或cell数组
                decs_raw = {Solutions.dec};  % 使用cell数组收集
                decs = cell2mat(decs_raw');  % 转换为矩阵
            catch ME
                % 如果提取失败，返回所有解（不过滤）
                Unique = Solutions;
                disp(['      [removeDuplicatesFast] Warning: Could not extract decs: ' ME.message ', returning all ', num2str(n), ' solutions']);
                return;
            end
            
            % 使用容差去重（避免浮点数精度问题）
            tolerance = 1e-4;  % 增大容差
            
            % 四舍五入到容差级别，然后使用unique
            decs_rounded = round(decs / tolerance) * tolerance;
            
            % 使用 unique 函数去重（基于行）
            [~, unique_idx] = unique(decs_rounded, 'rows', 'stable');
            Unique = Solutions(unique_idx);
            
            % 调试信息
            if length(Unique) < n
                disp(['      [removeDuplicatesFast] Removed ', num2str(n - length(Unique)), ' duplicates, kept ', num2str(length(Unique))]);
            end
        end
        
        % 保留：原始去重方法（兼容性）
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
        
        % 新增：基于非支配排序的选择（O(n log n)，替代 hypervolume）
        % 优化版本：增强多样性保持
        function Selected = selectByNDSort(obj, Solutions)
            n = length(Solutions);
            
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            
            % 第一步：非支配排序
            [FrontNo, ~] = NDSort(Solutions.objs, Solutions.cons, n);
            
            % 第二步：按前沿层选择
            Selected = [];
            current_front = 1;
            
            while length(Selected) < obj.MaxSize && current_front <= max(FrontNo)
                front_idx = find(FrontNo == current_front);
                front_solutions = Solutions(front_idx);
                
                if isempty(front_solutions)
                    current_front = current_front + 1;
                    continue;
                end
                
                % 如果这一层全部加入会超过限制，使用增强的多样性筛选
                if length(Selected) + length(front_solutions) > obj.MaxSize
                    remaining_slots = obj.MaxSize - length(Selected);
                    
                    if length(front_solutions) > remaining_slots
                        % 使用多样性增强的选择策略
                        front_solutions = obj.selectDiverseSolutions(front_solutions, remaining_slots);
                    end
                end
                
                Selected = [Selected, front_solutions];
                current_front = current_front + 1;
            end
        end
        
        % 新增：多样性增强的选择方法（修复版）
        function Selected = selectDiverseSolutions(obj, Solutions, n_select)
            n = length(Solutions);
            if n <= n_select
                Selected = Solutions;
                return;
            end
            
            objs = Solutions.objs;
            
            % 归一化目标值
            min_obj = min(objs, [], 1);
            max_obj = max(objs, [], 1);
            range_obj = max_obj - min_obj;
            range_obj(range_obj == 0) = 1;
            norm_objs = (objs - min_obj) ./ range_obj;
            
            % 方法：基于距离的最大化选择（Max-Min Diversity）
            Selected = false(1, n);
            
            % 首先选择极端点（每个目标的最小值和最大值点）
            % 修复：确保边界点被保留，提高多样性
            for m = 1:obj.M
                % 选择该目标的最小值点
                [~, min_idx] = min(objs(:, m));
                if ~Selected(min_idx)
                    Selected(min_idx) = true;
                end
                if sum(Selected) >= n_select
                    break;
                end
                
                % 选择该目标的最大值点
                [~, max_idx] = max(objs(:, m));
                if ~Selected(max_idx)
                    Selected(max_idx) = true;
                end
                if sum(Selected) >= n_select
                    break;
                end
            end
            
            % 然后使用贪心算法选择距离最远的点
            while sum(Selected) < n_select
                selected_idx = find(Selected);
                not_selected_idx = find(~Selected);
                
                if isempty(not_selected_idx)
                    break;
                end
                
                % 计算每个未选点到已选点集合的最小距离
                min_distances = inf(1, length(not_selected_idx));
                for i = 1:length(not_selected_idx)
                    idx = not_selected_idx(i);
                    % 计算到所有已选点的距离
                    distances = sqrt(sum((norm_objs(selected_idx, :) - norm_objs(idx, :)).^2, 2));
                    min_distances(i) = min(distances);
                end
                
                % 选择最小距离最大的点（Max-Min策略）
                [~, best_idx] = max(min_distances);
                Selected(not_selected_idx(best_idx)) = true;
            end
            
            Selected = Solutions(Selected);
        end
        
        % 新增：混合策略选择（根据Archive大小自动切换）
        function Selected = selectByHybridStrategy(obj, Solutions)
            n = length(Solutions);
            
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            
            % 混合策略：根据解集大小选择算法
            if n < obj.HVThreshold
                % 解集较小，使用hypervolume精确选择（高质量）
                disp(['      [HYBRID] Using Hypervolume selection (n=', num2str(n), ' < threshold=', num2str(obj.HVThreshold), ')']);
                Selected = obj.selectByHypervolume(Solutions);
            else
                % 解集较大，使用非支配排序快速选择（高速度）
                disp(['      [HYBRID] Using NDSort selection (n=', num2str(n), ' >= threshold=', num2str(obj.HVThreshold), ')']);
                Selected = obj.selectByNDSort(Solutions);
            end
        end
        
        % 保留：基于 hypervolume 的选择（精确但慢）
        function Selected = selectByHypervolume(obj, Solutions)
            n = length(Solutions);
            
            if n <= obj.MaxSize
                Selected = Solutions;
                return;
            end
            
            % 原始版本：O(n^2) 的 hypervolume 计算
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
            % 原始版本：2000 个采样点（高精度）
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
            % 优化：只返回非支配解（第一前沿）用于DM训练
            if isempty(obj.Solutions)
                Data = [];
                return;
            end
            
            total_count = length(obj.Solutions);
            
            % 如果解数量太少，直接返回所有解
            if total_count < 10
                Data = obj.Solutions;
                return;
            end
            
            % 非支配排序，只取第一前沿
            [FrontNo, ~] = NDSort(obj.Solutions.objs, obj.Solutions.cons, total_count);
            non_dominated = obj.Solutions(FrontNo == 1);
            non_dominated_count = length(non_dominated);
            
            % 如果非支配解太少，返回所有解确保训练数据充足
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
            
            % 如果当前档案大小超过新的最大值，进行剪枝（使用混合策略）
            if length(obj.Solutions) > new_max_size
                obj.Solutions = obj.selectByHybridStrategy(obj.Solutions);
            end
        end
        
        % 新增：获取待处理数量（用于调试）
        function count = getPendingCount(obj)
            count = length(obj.PendingSolutions);
        end
    end
end
