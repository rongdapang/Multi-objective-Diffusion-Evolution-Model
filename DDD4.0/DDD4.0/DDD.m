classdef DDD < ALGORITHM
% <2026> <multi> <real> <constrained/none>
% DDD: Dynamic Diffusion-Driven Multi-objective Optimization Algorithm
%
% This version uses the original hypervolume-based archive selection (SolutionArchive.m)
% which provides higher precision but slower performance (~17s per generation for Archive Add)
%
% Use this version when:
% - Solution quality is more important than speed
% - Archive size is small (<200)
% - Problem has 2-3 objectives (hypervolume calculation is faster)
%
% noise_schedule --- [0.1, 0.01] --- Noise schedule for diffusion model [start, end]
% network_structure --- [256, 512, 512, 256] --- Hidden layer structure for diffusion network
% ga_generations --- 20 --- Number of GA generations for initial sampling
% sample_size --- 500 --- Size of sample collection for training
% dm_epochs --- 100 --- Number of epochs for training diffusion model
% dm_steps --- 50 --- Number of diffusion steps for sampling
% archive_size --- 1000 --- Maximum size of elite solution archive
% update_interval --- 10 --- Generations between model updates
% dm_ratio --- 0.4 --- Base ratio of diffusion model offspring
% use_gpu --- true --- Whether to use GPU acceleration if available
% mating_pool_ratio --- 0.6 --- Base ratio of mating pool size (0.4-0.8 recommended)
% diversity_threshold --- 0.3 --- Threshold for diversity-based adjustment
%
%------------------------------- Reference --------------------------------
% Dynamic Diffusion-Driven Evolution for Multi-objective Optimization
% [Add your paper reference here when published]
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(Access = private)
        Archive         % SolutionArchive instance (using original hypervolume calculation)
        DMModel         % ConditionalDiffusionModel instance
        Scheduler       % AdaptiveScheduler instance
        Generation      % Current generation counter
        DMStats         % Statistics for DM offspring quality
        HasDLToolbox    % Flag indicating if Deep Learning Toolbox is available
        MatingPoolRatio % Current mating pool ratio
        DiversityHistory % History of population diversity
        BestObjHistory  % History of best objective values
        TimeStats       % Time statistics for each phase
        UseDynamicTournament % 是否使用动态锦标赛大小
        UseDeterministicDM   % 是否使用确定性DM回退
    end

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [noise_schedule, network_structure, ga_generations, sample_size, dm_epochs, dm_steps, archive_size, update_interval, dm_ratio, use_gpu, use_dynamic_tournament, use_deterministic_dm] = Algorithm.ParameterSet([0.1, 0.01], [256, 512, 512, 256], 20, 500, 100, 50, 1000, 10, 0.4, true, true, true);
            
            %% 检查消融实验全局变量（用于AblationStudy脚本）
            global ABLATION_TOURNAMENT ABLATION_DM;
            if ~isempty(ABLATION_TOURNAMENT)
                use_dynamic_tournament = ABLATION_TOURNAMENT;
            end
            if ~isempty(ABLATION_DM)
                use_deterministic_dm = ABLATION_DM;
            end
            
            %% Initialize components
            % Use SolutionArchive for original hypervolume-based selection
            Algorithm.Archive = SolutionArchive(archive_size, Problem.M);
            Algorithm.Scheduler = AdaptiveScheduler(dm_ratio, 0.15, 0.5);
            Algorithm.Generation = 0;
            Algorithm.DMStats = struct('total', 0, 'survived', 0, 'history', []);
            Algorithm.MatingPoolRatio = 0.6; % Optimized base ratio
            Algorithm.DiversityHistory = [];
            Algorithm.BestObjHistory = [];
            Algorithm.UseDynamicTournament = use_dynamic_tournament;
            Algorithm.UseDeterministicDM = use_deterministic_dm;
            
            %% Initialize time statistics
            Algorithm.TimeStats = struct(...
                'Stage1_InitialSampling', 0, ...
                'Stage2_DMTraining', 0, ...
                'Stage3_GenInitialSolutions', 0, ...
                'Stage4_GA', 0, ...
                'Stage4_DM', 0, ...
                'Stage4_EnvSelection', 0, ...
                'Stage4_ArchiveUpdate', 0, ...
                'Stage4_DMUpdate', 0, ...
                'TotalTime', 0, ...
                'GenTimes', []);
            
            TotalStart = tic;
            
            %% Check for Deep Learning Toolbox
            Algorithm.HasDLToolbox = Algorithm.checkDeepLearningToolbox();
            
            if ~Algorithm.HasDLToolbox
                warning('DDD:NoDLToolbox', 'Deep Learning Toolbox not found. Using enhanced GA-only mode.');
                use_gpu = false;
            end
            
            %% STAGE 1: High-quality initial sampling with GA
            disp('STAGE 1: Initial sampling with GA...');
            Stage1Start = tic;
            SamplePopulation = Algorithm.InitialSampling(Problem, ga_generations, sample_size);
            Algorithm.TimeStats.Stage1_InitialSampling = toc(Stage1Start);
            disp(['  - STAGE 1 Time: ', num2str(Algorithm.TimeStats.Stage1_InitialSampling, '%.2f'), ' seconds']);
            
            %% Add initial samples to archive
            ArchiveStart = tic;
            Algorithm.Archive.add(SamplePopulation);
            disp(['  - Archive Add Time: ', num2str(toc(ArchiveStart), '%.2f'), ' seconds']);
            
            %% STAGE 2: Train diffusion model
            Stage2Start = tic;
            if Algorithm.HasDLToolbox
                try
                    disp('STAGE 2: Training diffusion model...');
                    Algorithm.DMModel = ConditionalDiffusionModel(Problem.D, Problem.M, network_structure, noise_schedule, dm_steps, use_gpu);
                    Algorithm.DMModel = Algorithm.DMModel.train(SamplePopulation, Problem, dm_epochs);
                    
                    if ~Algorithm.DMModel.IsTrained
                        warning('DDD:DMTrainingFailed', 'Diffusion model training failed. Using GA-only mode.');
                        Algorithm.HasDLToolbox = false;
                        Algorithm.DMModel = [];
                    end
                catch ME
                    warning('DDD:DMInitError', ['Error initializing diffusion model: ' ME.message '. Using GA-only mode.']);
                    Algorithm.HasDLToolbox = false;
                    Algorithm.DMModel = [];
                end
            else
                Algorithm.DMModel = [];
            end
            Algorithm.TimeStats.Stage2_DMTraining = toc(Stage2Start);
            disp(['  - STAGE 2 Time: ', num2str(Algorithm.TimeStats.Stage2_DMTraining, '%.2f'), ' seconds']);
            
            %% STAGE 3: Generate initial solutions
            disp('STAGE 3: Generating initial solutions...');
            Stage3Start = tic;
            if Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                try
                    Population = Algorithm.GenerateInitialSolutions(Problem);
                catch ME
                    warning('DDD:DMGenError', ['Error generating initial solutions with DM: ' ME.message '. Using GA initialization.']);
                    Population = Algorithm.EnhancedGAInitialization(Problem, SamplePopulation);
                end
            else
                Population = Algorithm.EnhancedGAInitialization(Problem, SamplePopulation);
            end
            
            %% Add to archive
            ArchiveStart = tic;
            Algorithm.Archive.add(Population);
            Algorithm.TimeStats.Stage3_GenInitialSolutions = toc(Stage3Start);
            disp(['  - STAGE 3 Time: ', num2str(Algorithm.TimeStats.Stage3_GenInitialSolutions, '%.2f'), ' seconds']);
            disp(['  - Archive Add Time: ', num2str(toc(ArchiveStart), '%.2f'), ' seconds']);
            
            %% STAGE 4: Main optimization loop
            disp('STAGE 4: Main optimization loop...');
            while Algorithm.NotTerminated(Population)
                Algorithm.Generation = Algorithm.Generation + 1;
                GenStart = tic;
                
                %% Update mating pool ratio dynamically
                MatingPoolStart = tic;
                Algorithm.MatingPoolRatio = Algorithm.calculateMatingPoolRatio(Population, Problem);
                MatingPoolTime = toc(MatingPoolStart);
                
                %% Generate offspring with optimized mating pool
                OffspringStart = tic;
                [GAOffspring, DMOffspring, GATime, DMTime] = Algorithm.GenerateOffspring(Problem, Population);
                Algorithm.TimeStats.Stage4_GA = Algorithm.TimeStats.Stage4_GA + GATime;
                Algorithm.TimeStats.Stage4_DM = Algorithm.TimeStats.Stage4_DM + DMTime;
                OffspringTime = toc(OffspringStart);
                
                %% Update DM statistics
                DMStatsStart = tic;
                if ~isempty(DMOffspring)
                    Algorithm.UpdateDMStats(DMOffspring, Population);
                end
                DMStatsTime = toc(DMStatsStart);
                
                %% Environmental selection using NSGA-II mechanism
                EnvStart = tic;
                Combined = [Population, GAOffspring, DMOffspring];
                Population = DDD.EnvironmentalSelection(Combined, Problem.N);
                EnvTime = toc(EnvStart);
                Algorithm.TimeStats.Stage4_EnvSelection = Algorithm.TimeStats.Stage4_EnvSelection + EnvTime;
                
                %% Update archive with new solutions
                ArchiveStart = tic;
                % 优化：使用批量添加，每3代或Archive接近满时才完整处理
                % 修改：从每5代改为每3代，提高多样性维护频率
                if Algorithm.Generation <= 10 || mod(Algorithm.Generation, 3) == 0
                    % 前10代或每3代执行完整添加（包含剪枝和多样性选择）
                    Algorithm.Archive.add([GAOffspring, DMOffspring]);
                else
                    % 其他代使用批量缓存模式（内部也会维护多样性）
                    Algorithm.Archive.addBatch([GAOffspring, DMOffspring], Algorithm.Generation);
                end
                ArchiveTime = toc(ArchiveStart);
                Algorithm.TimeStats.Stage4_ArchiveUpdate = Algorithm.TimeStats.Stage4_ArchiveUpdate + ArchiveTime;
                
                %% Update diffusion model if needed
                DMUpdateStart = tic;
                if Algorithm.HasDLToolbox && Algorithm.ShouldUpdateModel(update_interval)
                    disp(['Updating diffusion model at generation ' num2str(Algorithm.Generation)]);
                    try
                        Algorithm.UpdateDiffusionModel(Problem, dm_epochs);
                    catch ME
                        warning('DDD:DMUpdateError', ['Error updating diffusion model: ' ME.message]);
                    end
                end
                DMUpdateTime = toc(DMUpdateStart);
                Algorithm.TimeStats.Stage4_DMUpdate = Algorithm.TimeStats.Stage4_DMUpdate + DMUpdateTime;
                
                %% Record generation time
                GenTotalTime = toc(GenStart);
                Algorithm.TimeStats.GenTimes = [Algorithm.TimeStats.GenTimes, GenTotalTime];
                
                %% Display detailed timing for this generation
                if mod(Algorithm.Generation, 5) == 0 || Algorithm.Generation <= 3
                    disp(['    === Gen ', num2str(Algorithm.Generation), ' Time Breakdown ===']);
                    disp(['      MatingPool Calc:   ', num2str(MatingPoolTime, '%.3f'), 's']);
                    disp(['      GA Operations:     ', num2str(GATime, '%.3f'), 's (', num2str(100*GATime/GenTotalTime, '%.1f'), '%)']);
                    disp(['      DM Operations:     ', num2str(DMTime, '%.3f'), 's (', num2str(100*DMTime/GenTotalTime, '%.1f'), '%)']);
                    disp(['      DM Stats Update:   ', num2str(DMStatsTime, '%.3f'), 's']);
                    disp(['      Env Selection:     ', num2str(EnvTime, '%.3f'), 's (', num2str(100*EnvTime/GenTotalTime, '%.1f'), '%)']);
                    disp(['      Archive Add:       ', num2str(ArchiveTime, '%.3f'), 's (', num2str(100*ArchiveTime/GenTotalTime, '%.1f'), '%)']);
                    disp(['      DM Model Update:   ', num2str(DMUpdateTime, '%.3f'), 's']);
                    disp(['      >>> GEN TOTAL:     ', num2str(GenTotalTime, '%.3f'), 's <<<']);
                    
                    % Diagnose crowding space every 5 generations
                    DDD.DiagnoseCrowdingSpace(Population, 0.1);
                end
            end
            
            %% Display final time statistics
            Algorithm.TimeStats.TotalTime = toc(TotalStart);
            Algorithm.displayTimeStats();
        end
        
        %% Calculate dynamic mating pool ratio
        function ratio = calculateMatingPoolRatio(Algorithm, Population, Problem)
            base_ratio = 0.6;
            
            total_generations = Problem.maxFE / Problem.N;
            current_ratio = Algorithm.Generation / total_generations;
            
            diversity = Algorithm.calculatePopulationDiversity(Population);
            Algorithm.DiversityHistory = [Algorithm.DiversityHistory, diversity];
            if length(Algorithm.DiversityHistory) > 20
                Algorithm.DiversityHistory = Algorithm.DiversityHistory(end-19:end);
            end
            
            ratio = base_ratio;
            
            if current_ratio < 0.3
                ratio = base_ratio * 0.7;
            elseif current_ratio < 0.6
                ratio = base_ratio * 0.85;
            else
                ratio = base_ratio;
            end
            
            if diversity < 0.3
                ratio = ratio * 1.2;
            elseif diversity > 0.7
                ratio = ratio * 0.85;
            end
            
            if length(Algorithm.DiversityHistory) >= 5
                diversity_trend = mean(Algorithm.DiversityHistory(end-2:end)) - mean(Algorithm.DiversityHistory(end-4:end-2));
                if diversity_trend < -0.05
                    ratio = ratio * 1.15;
                end
            end
            
            ratio = max(0.3, min(0.85, ratio));
        end
        
        %% Calculate population diversity
        function diversity = calculatePopulationDiversity(Algorithm, Population)
            if length(Population) < 2
                diversity = 0;
                return;
            end
            
            dec_matrix = Population.dec;
            max_dec = max(dec_matrix, [], 1);
            min_dec = max(dec_matrix, [], 1);
            range = max_dec - min_dec;
            range(range == 0) = 1;
            
            normalized_dec = (dec_matrix - min_dec) ./ range;
            
            mean_dec = mean(normalized_dec, 1);
            distances = sqrt(sum((normalized_dec - mean_dec).^2, 2));
            
            diversity = mean(distances) / sqrt(size(normalized_dec, 2));
            diversity = min(1, diversity);
        end
        
        %% Check if Deep Learning Toolbox is available
        function hasDL = checkDeepLearningToolbox(~)
            hasDL = false;
            try
                hasDL = license('test', 'Deep_Learning_Toolbox');
            catch
                hasDL = false;
            end
            
            if hasDL
                required_funcs = {'feedforwardnet', 'train', 'sim'};
                for i = 1:length(required_funcs)
                    if ~exist(required_funcs{i}, 'file')
                        hasDL = false;
                        return;
                    end
                end
            end
        end
        
        %% Enhanced GA-based initialization for fallback mode
        function Population = EnhancedGAInitialization(Algorithm, Problem, SamplePopulation)
            Population = Problem.Initialization();
            
            if ~isempty(SamplePopulation)
                [FrontNo, ~] = NDSort(SamplePopulation.objs, SamplePopulation.cons, length(SamplePopulation));
                ElitePop = SamplePopulation(FrontNo == 1);
                
                n_elite = min(length(ElitePop), floor(Problem.N * 0.3));
                if n_elite > 0
                    replace_idx = randperm(Problem.N, n_elite);
                    Population(replace_idx) = ElitePop(1:n_elite);
                end
            end
            
            n_mutate = floor(Problem.N * 0.2);
            mutate_idx = randperm(Problem.N, n_mutate);
            
            for i = 1:n_mutate
                parent = Population(mutate_idx(i)).dec;
                mutant = parent + 0.1 * randn(1, Problem.D) .* (Problem.upper - Problem.lower);
                mutant = max(min(mutant, Problem.upper), Problem.lower);
                Population(mutate_idx(i)) = Problem.Evaluation(mutant);
            end
        end
        
        %% Initial Sampling
        function SamplePopulation = InitialSampling(Algorithm, Problem, ga_generations, sample_size)
            Population = Problem.Initialization();
            
            for gen = 1:ga_generations
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                CrowdDis = CrowdingDistance(Population.objs, FrontNo);
                
                pool_size = max(10, floor(length(Population) * 0.5));
                MatingPool = TournamentSelection(2, pool_size, FrontNo, -CrowdDis);
                
                Offspring = OperatorGA(Problem, Population(MatingPool));
                Population = DDD.EnvironmentalSelection([Population, Offspring], Problem.N);
            end
            
            CurrentSize = length(Population);
            if CurrentSize < sample_size
                ExtraNeeded = sample_size - CurrentSize;
                ExtraDec = zeros(ExtraNeeded, Problem.D);
                
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                ElitePop = Population(FrontNo == 1);
                
                if isempty(ElitePop)
                    ElitePop = Population;
                end
                
                for i = 1:ExtraNeeded
                    parentIdx = randi(length(ElitePop));
                    parent = ElitePop(parentIdx).dec;
                    step_size = 0.1 * (1 - i/ExtraNeeded);
                    mutant = parent + step_size * randn(1, Problem.D) .* (Problem.upper - Problem.lower);
                    mutant = max(min(mutant, Problem.upper), Problem.lower);
                    ExtraDec(i, :) = mutant;
                end
                
                ExtraPop = Problem.Evaluation(ExtraDec);
                Population = [Population, ExtraPop];
            end
            
            SamplePopulation = Population;
        end
        
        %% Generate Initial Solutions
        function Population = GenerateInitialSolutions(Algorithm, Problem)
            n_solutions = Problem.N;
            TargetObjs = Algorithm.Archive.getTargetObjectives(n_solutions);
            Dec = Algorithm.DMModel.sample(TargetObjs);
            Dec = max(min(Dec, Problem.upper), Problem.lower);
            Population = Problem.Evaluation(Dec);
        end
        
        %% Generate Offspring with Optimized Mating Pool
        function [GAOffspring, DMOffspring, GATime, DMTime] = GenerateOffspring(Algorithm, Problem, Population)
            GAStart = tic;
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            CrowdDis = CrowdingDistance(Population.objs, FrontNo);
            
            pool_size = max(4, floor(Problem.N * Algorithm.MatingPoolRatio));
            
            % 方案2：根据进化进度动态调整锦标赛大小
            if Algorithm.UseDynamicTournament
                total_generations = Problem.maxFE / Problem.N;
                progress = Algorithm.Generation / total_generations;
                
                if progress < 0.3
                    tournament_size = 2;      % 前30%：探索期，保持多样性
                elseif progress < 0.7
                    tournament_size = 3;      % 30%-70%：平衡期，适度选择压力
                else
                    tournament_size = 4;      % 后30%：收敛期，增强选择压力
                end
            else
                tournament_size = 2;  % Baseline：固定为2
            end
            
            % 调试输出：每10代或阶段变化时显示锦标赛大小
            if mod(Algorithm.Generation, 10) == 1 || Algorithm.Generation <= 5
                total_generations = Problem.maxFE / Problem.N;
                progress = Algorithm.Generation / total_generations;
                mode_str = 'DYNAMIC';
                if ~Algorithm.UseDynamicTournament
                    mode_str = 'FIXED';
                end
                fprintf('    [Tournament-%s] Gen %d (%.1f%%): tournament_size = %d, pool_size = %d\n', ...
                    mode_str, Algorithm.Generation, progress*100, tournament_size, pool_size);
            end
            
            MatingPool = TournamentSelection(tournament_size, pool_size, FrontNo, -CrowdDis);
            GAOffspring = OperatorGA(Problem, Population(MatingPool));
            GATime = toc(GAStart);
            
            DMStart = tic;
            DMOffspring = [];
            if Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                dm_count = Algorithm.Scheduler.getDMOffspringCount(Problem.N, Algorithm.DMStats);
                
                if dm_count > 0
                    try
                        TargetObjs = Algorithm.selectTargetObjectives(Population, dm_count);
                        Dec = Algorithm.DMModel.sample(TargetObjs);
                        Dec = max(min(Dec, Problem.upper), Problem.lower);
                        DMOffspring = Problem.Evaluation(Dec);
                    catch ME
                        warning('DDD:DMSampleError', ['Error sampling from diffusion model: ' ME.message]);
                    end
                end
            end
            DMTime = toc(DMStart);
        end
        
        %% Select Target Objectives
        function TargetObjs = selectTargetObjectives(Algorithm, Population, n_targets)
            PopObjs = Population.objs;
            ArchiveObjs = Algorithm.Archive.getObjectives();
            
            if ~isempty(ArchiveObjs)
                AllObjs = [PopObjs; ArchiveObjs];
            else
                AllObjs = PopObjs;
            end
            
            min_obj = min(AllObjs, [], 1);
            max_obj = max(AllObjs, [], 1);
            range_obj = max_obj - min_obj;
            range_obj(range_obj == 0) = 1;
            
            NormObjs = (AllObjs - min_obj) ./ range_obj;
            
            if size(NormObjs, 1) <= n_targets
                TargetObjs = AllObjs;
            else
                try
                    [~, C] = kmeans(NormObjs, n_targets, 'MaxIter', 100);
                    TargetObjs = C .* range_obj + min_obj;
                catch
                    if Algorithm.UseDeterministicDM
                        % 改进：使用确定性采样替代随机选择
                        TargetObjs = DDD.DeterministicTargetSelection(AllObjs, NormObjs, n_targets, range_obj, min_obj);
                    else
                        % Baseline：随机选择
                        idx = randperm(size(AllObjs, 1), n_targets);
                        TargetObjs = AllObjs(idx, :);
                    end
                end
            end
        end
        
        %% Deterministic Target Selection (replacement for random fallback)
        function TargetObjs = DeterministicTargetSelection(AllObjs, NormObjs, n_targets, range_obj, min_obj)
            n = size(AllObjs, 1);
            
            % 策略1：尝试使用网格划分选择
            % 将目标空间划分为n_targets个区域，从每个区域选一个代表
            
            % 计算每个解的综合得分（考虑各目标的平衡）
            % 使用加权和来评估解在目标空间中的位置
            M = size(NormObjs, 2);
            
            % 为每个解计算一个标量值，用于排序和选择
            % 使用各目标的加权和，权重基于该解在各目标上的相对表现
            scores = zeros(n, 1);
            for i = 1:n
                % 计算该解在归一化目标空间中的"位置"
                % 优先选择边界和极端点
                obj_vals = NormObjs(i, :);
                % 得分 = 到原点的距离 + 到各极端方向的接近程度
                dist_to_origin = norm(obj_vals);
                min_obj_val = min(obj_vals);
                scores(i) = dist_to_origin + (1 - min_obj_val);  % 优先选择边界解
            end
            
            % 按得分排序
            [~, sorted_idx] = sort(scores, 'descend');
            
            % 策略2：均匀间隔选择
            % 从排序后的列表中均匀采样，确保覆盖不同区域
            if n >= n_targets
                step = floor(n / n_targets);
                selected_idx = sorted_idx(1:step:min(n, step*n_targets));
                % 如果选的不够，补充剩余的最佳解
                if length(selected_idx) < n_targets
                    remaining = setdiff(sorted_idx, selected_idx);
                    n_needed = n_targets - length(selected_idx);
                    selected_idx = [selected_idx; remaining(1:min(n_needed, length(remaining)))'];
                end
            else
                selected_idx = sorted_idx;
            end
            
            % 确保选中的目标数量正确
            selected_idx = selected_idx(1:min(n_targets, length(selected_idx)));
            
            TargetObjs = AllObjs(selected_idx, :);
            
            % 调试输出
            fprintf('    [DM Target Selection] kmeans failed, using deterministic selection\n');
            fprintf('    [DM Target Selection] Selected %d targets from %d candidates\n', length(selected_idx), n);
        end
        
        %% Update DM Statistics
        function UpdateDMStats(Algorithm, DMOffspring, Population)
            if isempty(DMOffspring)
                return;
            end
            
            % 修复：使用容忍度比较浮点数，而不是 isequal
            % 原代码：isequal(DMOffspring(i).dec, Population(j).dec) 几乎永远为 false
            tolerance = 1e-6;
            n_survived = 0;
            for i = 1:length(DMOffspring)
                for j = 1:length(Population)
                    if all(abs(DMOffspring(i).dec - Population(j).dec) < tolerance)
                        n_survived = n_survived + 1;
                        break;
                    end
                end
            end
            
            Algorithm.DMStats.total = Algorithm.DMStats.total + length(DMOffspring);
            Algorithm.DMStats.survived = Algorithm.DMStats.survived + n_survived;
            
            survival_rate = n_survived / length(DMOffspring);
            Algorithm.DMStats.history = [Algorithm.DMStats.history, survival_rate];
            
            if length(Algorithm.DMStats.history) > 50
                Algorithm.DMStats.history = Algorithm.DMStats.history(end-49:end);
            end
        end
        
        %% Should Update Model
        function shouldUpdate = ShouldUpdateModel(Algorithm, update_interval)
            shouldUpdate = false;
            
            if mod(Algorithm.Generation, update_interval) == 0
                shouldUpdate = true;
                return;
            end
            
            if ~isempty(Algorithm.DMStats.history) && length(Algorithm.DMStats.history) >= 5
                recent_performance = mean(Algorithm.DMStats.history(end-4:end));
                if recent_performance < 0.1
                    shouldUpdate = true;
                end
            end
        end
        
        %% Update Diffusion Model
        function UpdateDiffusionModel(Algorithm, Problem, dm_epochs)
            if isempty(Algorithm.DMModel) || ~Algorithm.DMModel.IsTrained
                return;
            end
            
            TrainingData = Algorithm.Archive.getTrainingData();
            
            if length(TrainingData) < 10
                warning('DDD:InsufficientData', 'Insufficient data for model update');
                return;
            end
            
            try
                Algorithm.DMModel = Algorithm.DMModel.train(TrainingData, Problem, dm_epochs);
            catch ME
                warning('DDD:DMRetrainError', ['Error retraining diffusion model: ' ME.message]);
            end
        end
        
        %% Display Time Statistics
        function displayTimeStats(Algorithm)
            disp(' ');
            disp('╔════════════════════════════════════════════════════════════╗');
            disp('║                 DDD Time Statistics                        ║');
            disp('╠════════════════════════════════════════════════════════════╣');
            
            ts = Algorithm.TimeStats;
            
            % Stage times
            disp('║  STAGE BREAKDOWN                                           ║');
            disp(['║    STAGE 1 - Initial Sampling:    ', sprintf('%8.2f', ts.Stage1_InitialSampling), 's                    ║']);
            disp(['║    STAGE 2 - DM Training:         ', sprintf('%8.2f', ts.Stage2_DMTraining), 's                    ║']);
            disp(['║    STAGE 3 - Gen Initial Sol:     ', sprintf('%8.2f', ts.Stage3_GenInitialSolutions), 's                    ║']);
            disp('║                                                            ║');
            
            % Stage 4 breakdown
            disp('║  STAGE 4 - Main Loop (per generation averages)             ║');
            n_gen = length(ts.GenTimes);
            if n_gen > 0
                disp(['║    Number of Generations:         ', sprintf('%8d', n_gen), '                         ║']);
                disp(['║    Avg Time per Generation:       ', sprintf('%8.3f', mean(ts.GenTimes)), 's                    ║']);
                disp('║                                                            ║');
                disp(['║    GA Operations (total):         ', sprintf('%8.2f', ts.Stage4_GA), 's                    ║']);
                disp(['║    DM Operations (total):         ', sprintf('%8.2f', ts.Stage4_DM), 's                    ║']);
                disp(['║    Env Selection (total):         ', sprintf('%8.2f', ts.Stage4_EnvSelection), 's                    ║']);
                disp(['║    Archive Update (total):        ', sprintf('%8.2f', ts.Stage4_ArchiveUpdate), 's                    ║']);
                disp(['║    DM Model Update (total):       ', sprintf('%8.2f', ts.Stage4_DMUpdate), 's                    ║']);
            end
            
            disp('║                                                            ║');
            disp(['║  TOTAL TIME:                      ', sprintf('%8.2f', ts.TotalTime), 's                    ║']);
            disp('╚════════════════════════════════════════════════════════════╝');
            disp(' ');
            
            % Generation time trend
            if n_gen >= 10
                disp('  Generation Time Trend (last 10 gens):');
                early_avg = mean(ts.GenTimes(1:min(5, n_gen)));
                late_avg = mean(ts.GenTimes(max(1, n_gen-4):n_gen));
                disp(['    Early generations avg: ', sprintf('%.3f', early_avg), 's']);
                disp(['    Late generations avg:  ', sprintf('%.3f', late_avg), 's']);
                if late_avg > early_avg * 1.5
                    disp('    ⚠️  Warning: Generation time increased significantly!');
                end
                disp(' ');
            end
        end
    end
    
    methods(Static)
        %% Environmental Selection
        function Population = EnvironmentalSelection(Population, N)
            [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
            Next = FrontNo < MaxFNo;
            
            Last = find(FrontNo == MaxFNo);
            if sum(Next) + length(Last) > N
                CrowdDis = CrowdingDistance(Population(Last).objs, ones(1, length(Last)));
                [~, Rank] = sort(CrowdDis, 'descend');
                Next(Last(Rank(1:N-sum(Next)))) = true;
            else
                Next(Last) = true;
            end
            
            Population = Population(Next);
        end
        
        %% Calculate Objective Space Diversity
        function diversity = ObjectiveSpaceDiversity(Population)
            objs = Population.objs;
            n = size(objs, 1);
            
            if n < 2
                diversity = 0;
                return;
            end
            
            % Normalize objectives
            min_obj = min(objs, [], 1);
            max_obj = max(objs, [], 1);
            range_obj = max_obj - min_obj;
            range_obj(range_obj == 0) = 1;
            norm_objs = (objs - min_obj) ./ range_obj;
            
            % Calculate pairwise distances (sample for efficiency if too many)
            if n > 100
                % Sample 100 pairs for efficiency
                sample_size = 100;
                total_dist = 0;
                for k = 1:sample_size
                    i = randi(n);
                    j = randi(n);
                    while j == i
                        j = randi(n);
                    end
                    dist = norm(norm_objs(i,:) - norm_objs(j,:));
                    total_dist = total_dist + dist;
                end
                diversity = total_dist / sample_size;
            else
                % Calculate all pairwise distances
                total_dist = 0;
                count = 0;
                for i = 1:n
                    for j = i+1:n
                        dist = norm(norm_objs(i,:) - norm_objs(j,:));
                        total_dist = total_dist + dist;
                        count = count + 1;
                    end
                end
                diversity = total_dist / count;
            end
        end
        
        %% Calculate Decision Space Diversity
        function diversity = DecisionSpaceDiversity(Population)
            decs = Population.dec;
            n = size(decs, 1);
            
            if n < 2
                diversity = 0;
                return;
            end
            
            % Debug: print raw stats
            min_dec = min(decs, [], 1);
            max_dec = max(decs, [], 1);
            range_dec = max_dec - min_dec;
            fprintf('    [DEBUG] decs size: %dx%d, range: [%.6f, %.6f], mean range: %.6f\n', ...
                n, size(decs, 2), min(range_dec), max(range_dec), mean(range_dec));
            
            % Normalize decision variables
            range_dec(range_dec == 0) = 1;
            norm_decs = (decs - min_dec) ./ range_dec;
            
            % Check if all normalized values are the same
            norm_range = max(norm_decs, [], 1) - min(norm_decs, [], 1);
            fprintf('    [DEBUG] norm_decs range: [%.6f, %.6f], mean: %.6f\n', ...
                min(norm_range), max(norm_range), mean(norm_range));
            
            % Calculate pairwise distances (sample for efficiency if too many)
            if n > 100
                % Sample 100 pairs for efficiency
                sample_size = 100;
                total_dist = 0;
                for k = 1:sample_size
                    i = randi(n);
                    j = randi(n);
                    while j == i
                        j = randi(n);
                    end
                    dist = norm(norm_decs(i,:) - norm_decs(j,:));
                    total_dist = total_dist + dist;
                end
                diversity = total_dist / sample_size;
            else
                % Calculate all pairwise distances
                total_dist = 0;
                count = 0;
                for i = 1:n
                    for j = i+1:n
                        dist = norm(norm_decs(i,:) - norm_decs(j,:));
                        total_dist = total_dist + dist;
                        count = count + 1;
                    end
                end
                if count > 0
                    diversity = total_dist / count;
                else
                    diversity = 0;
                end
            end
            fprintf('    [DEBUG] total_dist: %.6f, count: %d, diversity: %.6f\n', ...
                total_dist, count, diversity);
        end
        
        %% Diagnose Crowding Space
        function DiagnoseCrowdingSpace(Population, threshold)
            if nargin < 2
                threshold = 0.1;  % Default threshold
            end
            
            obj_div = DDD.ObjectiveSpaceDiversity(Population);
            dec_div = DDD.DecisionSpaceDiversity(Population);
            
            % Additional debug info for decision space
            decs = Population.dec;
            dec_range = max(decs, [], 1) - min(decs, [], 1);
            dec_std = std(decs, 0, 1);
            
            fprintf('\n  === Diversity Diagnosis ===\n');
            fprintf('    Objective Space Diversity: %.4f', obj_div);
            if obj_div < threshold
                fprintf('  [WARNING: Below threshold %.2f]\n', threshold);
            else
                fprintf('  [OK]\n');
            end
            
            fprintf('    Decision Space Diversity: %.4f', dec_div);
            if dec_div < threshold
                fprintf('  [WARNING: Below threshold %.2f]\n', threshold);
            else
                fprintf('  [OK]\n');
            end
            
            % Debug: show decision variable statistics
            fprintf('    Decision Variables Stats:\n');
            fprintf('      - Number of vars: %d\n', size(decs, 2));
            fprintf('      - Range (max-min): [%.4f, %.4f], mean=%.4f\n', min(dec_range), max(dec_range), mean(dec_range));
            fprintf('      - Std dev: [%.4f, %.4f], mean=%.4f\n', min(dec_std), max(dec_std), mean(dec_std));
            fprintf('      - Unique rows: %d / %d\n', size(unique(decs, 'rows'), 1), size(decs, 1));
            
            % Determine crowding type
            fprintf('    Diagnosis: ');
            if obj_div < threshold && dec_div < threshold
                fprintf('Crowding in BOTH spaces\n');
            elseif obj_div < threshold
                fprintf('Crowding in OBJECTIVE space\n');
            elseif dec_div < threshold
                fprintf('Crowding in DECISION space\n');
            else
                fprintf('Diversity is good in both spaces\n');
            end
            fprintf('  =============================\n\n');
        end
    end
end
