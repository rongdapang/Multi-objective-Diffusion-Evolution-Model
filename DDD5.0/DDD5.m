classdef DDD5 < ALGORITHM
% <2026> <multi> <real> <constrained/none>
% DDD5: Dynamic Diffusion-Driven Multi-objective Optimization Algorithm
% This version uses the original hypervolume-based archive selection (SolutionArchive.m)
% which provides higher precision but slower performance (~17s per generation for Archive Add)
% Use this version when:
% - Solution quality is more important than speed
% - Archive size is small (<200)
% - Problem has 2-3 objectives (hypervolume calculation is faster)
% noise_schedule --- [0.1, 0.01] --- Noise schedule for diffusion model [start, end]
% network_structure --- [512, 1024, 1024, 512, 256] --- Hidden layer structure for diffusion network
% sample_size --- 1200 --- Size of sample collection for training
% dm_epochs --- 250 --- Number of epochs for training diffusion model
% dm_steps --- 150 --- Number of diffusion steps for sampling
% archive_size --- 3000 --- Maximum size of elite solution archive
% update_interval --- 10 --- Generations between model updates
% dm_ratio --- 0.4 --- Base ratio of diffusion model offspring
% use_gpu --- true --- Whether to use GPU acceleration if available
% use_dynamic_tournament --- true --- Whether to use dynamic tournament size
% use_deterministic_dm --- true --- Whether to use deterministic DM fallback
% min_nd_solutions --- 200 --- Minimum non-dominated solutions to start diffusion
% max_initial_generations --- 150 --- Maximum generations for initial phase
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
        Archive                 % SolutionArchive instance
        DMModel                 % ConditionalDiffusionModel instance
        Scheduler               % AdaptiveScheduler instance
        Generation              % Current generation counter
        DMStats                 % Statistics for DM offspring quality
        HasDLToolbox            % Flag indicating if Deep Learning Toolbox is available
        MatingPoolRatio         % Current mating pool ratio
        DiversityHistory        % History of population diversity
        BestObjHistory          % History of best objective values
        TimeStats               % Time statistics for each phase
        UseDynamicTournament    % 是否使用动态锦标赛大小
        UseDeterministicDM      % 是否使用确定性 DM 回退
        MinNDSolutions          % 启动扩散所需的最小非支配解数量
        MaxInitialGenerations   % 初始阶段最大代数
    end

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [noise_schedule, network_structure, sample_size, dm_epochs, dm_steps, archive_size, update_interval, dm_ratio, use_gpu, use_dynamic_tournament, use_deterministic_dm, min_nd_solutions, max_initial_generations] = ...
                Algorithm.ParameterSet([0.1, 0.01], [512, 1024, 1024, 512, 256], 1200, 250, 150, 3000, 10, 0.4, true, true, true, 200, 300);
            
            % 检查消融实验全局变量（用于 AblationStudy 脚本）
            global ABLATION_TOURNAMENT ABLATION_DM;
        if ~isempty(ABLATION_TOURNAMENT)
            use_dynamic_tournament = ABLATION_TOURNAMENT;
        end
        if ~isempty(ABLATION_DM)
            use_deterministic_dm = ABLATION_DM;
        end
        
        %% Adaptive parameter adjustment for high-dimensional problems
        isHighDimMain = Problem.D >= 50;
        isMultimodalMain = Algorithm.isMultimodalProblem(Problem);
        if isHighDimMain || isMultimodalMain
            % === ULTRA-HIGH QUALITY MODE ===
            max_initial_generations = min(max_initial_generations * 3, 500);
            % 高维问题：降低 ND 门槛（因为 ND 解本来就少）
            min_nd_solutions = floor(min_nd_solutions * 0.8);
            % Store high quality mode flag for later use
            high_quality_mode = true;
            disp('╔════════════════════════════════════════════════════════════╗');
            disp('║  ULTRA-HIGH QUALITY MODE ACTIVATED                         ║');
            disp(['║  Dimension: ', sprintf('%3d', Problem.D), '                                            ║']);
            disp(['║  max_initial_generations: ', sprintf('%3d', max_initial_generations), ' (x3, cap=500)                     ║']);
            disp(['║  min_nd_solutions: ', sprintf('%3d', min_nd_solutions), ' (x0.8 for high-dim)                ║']);
            disp('╚════════════════════════════════════════════════════════════╝');
        else
            high_quality_mode = false;
        end
        
        %% Initialize components
        % Use SolutionArchive for original hypervolume-based selection
        try
            Algorithm.Archive = FastArchive(min(archive_size, 500), Problem.M);
            disp('  [Archive] Using FastArchive for better performance');
        catch
            Algorithm.Archive = SolutionArchive(min(archive_size, 300), Problem.M);
            disp('  [Archive] Using SolutionArchive with reduced size (300)');
        end
        
        Algorithm.Scheduler = AdaptiveScheduler(dm_ratio, 0.15, 0.5);
        Algorithm.Generation = 0;
        Algorithm.DMStats = struct('total', 0, 'survived', 0, 'history', []);
        Algorithm.MatingPoolRatio = 0.6;
        Algorithm.DiversityHistory = [];
        Algorithm.BestObjHistory = [];
        Algorithm.UseDynamicTournament = use_dynamic_tournament;
        Algorithm.UseDeterministicDM = use_deterministic_dm;
        Algorithm.MinNDSolutions = min_nd_solutions;
        Algorithm.MaxInitialGenerations = max_initial_generations;
        
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
        
        %% STAGE 1: Quality-driven initial evolution (adaptive generations)
        disp('╔════════════════════════════════════════════════════════════╗');
        disp('║  STAGE 1: Quality-Driven Initial Evolution                 ║');
        disp('╚════════════════════════════════════════════════════════════╝');
        Stage1Start = tic;
        [SamplePopulation, initial_generations] = Algorithm.QualityDrivenInitialEvolution(Problem, sample_size, min_nd_solutions, max_initial_generations);
        Algorithm.TimeStats.Stage1_InitialSampling = toc(Stage1Start);
        disp(['  ✓ Initial evolution completed after ', num2str(initial_generations), ' generations']);
        disp(['  ✓ STAGE 1 Time: ', num2str(Algorithm.TimeStats.Stage1_InitialSampling, '%.2f'), ' seconds']);
        
        %% Add initial samples to archive
        ArchiveStart = tic;
        Algorithm.Archive.add(SamplePopulation);
        disp(['  ✓ Archive initialized with ', num2str(length(SamplePopulation)), ' solutions']);
        disp(['  ✓ Archive Add Time: ', num2str(toc(ArchiveStart), '%.2f'), ' seconds']);
        
        %% STAGE 2: Train diffusion model (only if quality is sufficient)
        Stage2Start = tic;
        if Algorithm.HasDLToolbox
            try
                % Check if we have enough quality data for training
                TrainingData = Algorithm.Archive.getTrainingData();
                [quality_score, nd_count, obj_range, quality_details] = Algorithm.evaluateArchiveQuality(Problem);
                
                % Get objectives for display
                if ~isempty(TrainingData)
                    objs_display = TrainingData.objs;
                    max_f1 = max(objs_display(:, 1));
                else
                    max_f1 = 0;
                end
                
                disp(' ');
                disp('╔════════════════════════════════════════════════════════════╗');
                disp('║  STAGE 2: Training diffusion model...                      ║');
                disp('╚════════════════════════════════════════════════════════════╝');
                disp(['  - Non-dominated solutions: ', num2str(nd_count)]);
                disp(['  - Quality score: ', num2str(quality_score, '%.4f')]);
                disp(['  - Objective range: ', num2str(obj_range, '%.4f')]);
                disp(['  - Proximity factor: ', num2str(quality_details.proximity_factor, '%.4f')]);
                disp(['  - Obj range: f1=[', num2str(quality_details.min_f1, '%.3f'), ',', num2str(max_f1, '%.3f'), ']']);
                disp(['  - Obj range: f2=[', num2str(quality_details.min_f2, '%.3f'), ',', num2str(quality_details.max_f2, '%.3f'), ']']);
                
                % Check BOTH non-dominated count AND proximity
                % === SMART QUALITY THRESHOLDS ===
                % Key insight: When proximity is high (solutions near PF), fewer ND solutions is acceptable
                % because solutions are concentrated near the true front
                
                proximity = quality_details.proximity_factor;
                
                if high_quality_mode
                    if Problem.D >= 50
                        min_nd_absolute = 30;
                        min_proximity = 0.5;
                        disp('  [Ultra-High Quality - HighDim] Thresholds: min_nd=30, proximity=0.5');
                    else
                        min_nd_absolute = 50;
                        min_proximity = 0.5;
                        disp('  [Ultra-High Quality] Thresholds: min_nd=50, proximity=0.5');
                    end
                else
                    min_nd_absolute = 20;
                    min_proximity = 0.3;
                end
                
                % SMART THRESHOLD: Lower ND requirement when proximity is high
                % If proximity >= 0.8, solutions are very close to PF, accept fewer ND
                if proximity >= 0.8
                    nd_threshold = max(30, min_nd_absolute * 0.5);
                    disp(['  [Smart Threshold] High proximity (', num2str(proximity, '%.2f'), '), lowering ND requirement to ', num2str(nd_threshold)]);
                elseif proximity >= 0.5
                    nd_threshold = max(40, min_nd_absolute * 0.7);
                    disp(['  [Smart Threshold] Good proximity (', num2str(proximity, '%.2f'), '), ND requirement = ', num2str(nd_threshold)]);
                else
                    nd_threshold = min_nd_absolute;
                end
                
                if nd_count >= nd_threshold && proximity >= min_proximity
                    Algorithm.DMModel = ConditionalDiffusionModel(Problem.D, Problem.M, network_structure, noise_schedule, dm_steps, use_gpu);
                    Algorithm.DMModel = Algorithm.DMModel.train(TrainingData, Problem, dm_epochs);
                    if ~Algorithm.DMModel.IsTrained
                        warning('DDD:DMTrainingFailed', 'Diffusion model training failed. Using GA-only mode.');
                        Algorithm.HasDLToolbox = false;
                        Algorithm.DMModel = [];
                    else
                        disp('  ✓ Diffusion model trained successfully');
                    end
                elseif quality_details.proximity_factor < 0.3
                    warning('DDD:InsufficientQuality', sprintf('Solutions too far from PF (proximity=%.3f). Using GA-only mode.', quality_details.proximity_factor));
                    Algorithm.HasDLToolbox = false;
                    Algorithm.DMModel = [];
                else
                    warning('DDD:InsufficientQuality', ['Insufficient non-dominated solutions (' num2str(nd_count) '). Using GA-only mode.']);
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
        disp(['  ✓ STAGE 2 Time: ', num2str(Algorithm.TimeStats.Stage2_DMTraining, '%.2f'), ' seconds']);
        
        %% STAGE 3: Generate Initial Solutions
        Stage3Start = tic;
        if Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
            try
                Population = Algorithm.GenerateInitialSolutions(Problem);
                disp('  ✓ Initial solutions generated using diffusion model');
            catch ME
                warning('DDD:DMGenError', ['Error generating initial solutions with DM: ' ME.message '. Using GA initialization.']);
                Population = Algorithm.EnhancedGAInitialization(Problem, SamplePopulation);
                disp('  ✓ Initial solutions generated using GA (fallback)');
            end
        else
            Population = Algorithm.EnhancedGAInitialization(Problem, SamplePopulation);
            disp('  ✓ Initial solutions generated using GA');
        end
        
        % Add to archive
        ArchiveStart = tic;
        Algorithm.Archive.add(Population);
        Algorithm.TimeStats.Stage3_GenInitialSolutions = toc(Stage3Start);
        disp(['  ✓ STAGE 3 Time: ', num2str(Algorithm.TimeStats.Stage3_GenInitialSolutions, '%.2f'), ' seconds']);
        disp(['  ✓ Archive Add Time: ', num2str(toc(ArchiveStart), '%.2f'), ' seconds']);
        
        %% STAGE 4: Main optimization loop
        disp('STAGE 4: Main optimization loop...');
        while Algorithm.NotTerminated(Population)
            Algorithm.Generation = Algorithm.Generation + 1;
            GenStart = tic;
            
            % Update mating pool ratio dynamically
            MatingPoolStart = tic;
            Algorithm.MatingPoolRatio = Algorithm.calculateMatingPoolRatio(Population, Problem);
            MatingPoolTime = toc(MatingPoolStart);
            
            % Generate offspring with optimized mating pool
            OffspringStart = tic;
            [GAOffspring, DMOffspring, GATime, DMTime] = Algorithm.GenerateOffspring(Problem, Population);
            Algorithm.TimeStats.Stage4_GA = Algorithm.TimeStats.Stage4_GA + GATime;
            Algorithm.TimeStats.Stage4_DM = Algorithm.TimeStats.Stage4_DM + DMTime;
            OffspringTime = toc(OffspringStart);
            
            % Update DM statistics
            DMStatsStart = tic;
            if ~isempty(DMOffspring)
                Algorithm.UpdateDMStats(DMOffspring, Population);
            end
            DMStatsTime = toc(DMStatsStart);
            
            % Environmental selection using NSGA-II mechanism
            EnvStart = tic;
            Combined = [Population, GAOffspring, DMOffspring];
            Population = DDD5.EnvironmentalSelection(Combined, Problem.N);
            EnvTime = toc(EnvStart);
            Algorithm.TimeStats.Stage4_EnvSelection = Algorithm.TimeStats.Stage4_EnvSelection + EnvTime;
            
            % Update archive with new solutions
            ArchiveStart = tic;
            % 优化：使用批量添加，每 3 代或 Archive 接近满时才完整处理
            if Algorithm.Generation <= 10 || mod(Algorithm.Generation, 3) == 0
                % 前 10 代或每 3 代执行完整添加（包含剪枝和多样性选择）
                Algorithm.Archive.add([GAOffspring, DMOffspring]);
            else
                % 其他代使用批量缓存模式（内部也会维护多样性）
                Algorithm.Archive.addBatch([GAOffspring, DMOffspring], Algorithm.Generation);
            end
            ArchiveTime = toc(ArchiveStart);
            Algorithm.TimeStats.Stage4_ArchiveUpdate = Algorithm.TimeStats.Stage4_ArchiveUpdate + ArchiveTime;
            
            % Update diffusion model if needed
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
            
            % Record generation time
            GenTotalTime = toc(GenStart);
            Algorithm.TimeStats.GenTimes = [Algorithm.TimeStats.GenTimes, GenTotalTime];
            
            %% Display detailed timing for this generation
            if mod(Algorithm.Generation, 5) == 0 || Algorithm.Generation <= 3
                disp(['    === Gen ', num2str(Algorithm.Generation), ' Time Breakdown ===']);
                disp(['      MatingPool Calc:   ', num2str(MatingPoolTime, '%.3f'), 's']);
                disp(['      GA Operations:     ', num2str(GATime, '%.3f'), 's (', num2str(100 * GATime / GenTotalTime, '%.1f'), '%)']);
                disp(['      DM Operations:     ', num2str(DMTime, '%.3f'), 's (', num2str(100 * DMTime / GenTotalTime, '%.1f'), '%)']);
                disp(['      DM Stats Update:   ', num2str(DMStatsTime, '%.3f'), 's']);
                disp(['      Env Selection:     ', num2str(EnvTime, '%.3f'), 's (', num2str(100 * EnvTime / GenTotalTime, '%.1f'), '%)']);
                disp(['      Archive Add:       ', num2str(ArchiveTime, '%.3f'), 's (', num2str(100 * ArchiveTime / GenTotalTime, '%.1f'), '%)']);
                disp(['      DM Model Update:   ', num2str(DMUpdateTime, '%.3f'), 's']);
                disp(['      >>> GEN TOTAL:     ', num2str(GenTotalTime, '%.3f'), 's <<<']);
                % Diagnose crowding space every 5 generations
                DDD5.DiagnoseCrowdingSpace(Population, 0.1);
            end
            
            % Display final time statistics
            Algorithm.TimeStats.TotalTime = toc(TotalStart);
            Algorithm.displayTimeStats();
        end
    end

    % Calculate dynamic mating pool ratio
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

    % Calculate population diversity
    function diversity = calculatePopulationDiversity(Algorithm, Population)
        if length(Population) < 2
            diversity = 0;
            return;
        end
        dec_matrix = Population.dec;
        max_dec = max(dec_matrix, [], 1);
        min_dec = min(dec_matrix, [], 1);
        range = max_dec - min_dec;
        range(range == 0) = 1;
        normalized_dec = (dec_matrix - min_dec) ./ range;
        mean_dec = mean(normalized_dec, 1);
        distances = sqrt(sum((normalized_dec - mean_dec).^2, 2));
        diversity = mean(distances) / sqrt(size(normalized_dec, 2));
        diversity = min(1, diversity);
    end

    % Check if Deep Learning Toolbox is available
    function hasDL = checkDeepLearningToolbox(~)
        hasDL = false;
        try
            testLayer = fullyConnectedLayer(10);
            layers = [imageInputLayer([1 1 1]); fullyConnectedLayer(1)];
            net = dlnetwork(layers);
            hasDL = true;
            clear testLayer layers net;
        catch
            hasDL = false;
        end
    end
end

methods
    %% Quality-Driven Initial Evolution (replaces fixed-ga_generations approach)
    function [SamplePopulation, total_generations] = QualityDrivenInitialEvolution(Algorithm, Problem, sample_size, min_nd_solutions, max_initial_generations)
        % Quality-driven initial evolution that adapts until sufficient non-dominated solutions are found
        isHighDim = Problem.D >= 50;
        isMultimodal = Algorithm.isMultimodalProblem(Problem);
        if isHighDim
            disp('  [HIGH-DIM] Detected high-dimensional problem (D >= 50)');
            disp(['  [HIGH-DIM] Using specialized initialization for D=', num2str(Problem.D)]);
        end
        
        % Use enhanced initialization for high-dim/multimodal problems
        if isHighDim || isMultimodal
            Population = Algorithm.EnhancedHighDimInitialization(Problem);
        else
            Population = Problem.Initialization();
        end
        Algorithm.Archive.add(Population);
        
        total_generations = 0;
        quality_history = [];
        nd_count_history = [];
        disp(' ');
        disp('  Starting quality-driven evolution...');
        disp(['  Target: ', num2str(min_nd_solutions), ' non-dominated solutions']);
        disp(['  Maximum generations: ', num2str(max_initial_generations)]);
        disp(' ');
        
        while total_generations < max_initial_generations
            total_generations = total_generations + 1;
            gen_start = tic;
            
            % Adaptive GA evolution based on problem characteristics
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            CrowdDis = CrowdingDistance(Population.objs, FrontNo);
            
            % High-dim problems: use larger mating pool and tournament
            if isHighDim
                pool_size = max(20, floor(length(Population) * 0.7));
                tournament_size = 3;
            else
                pool_size = max(10, floor(length(Population) * 0.5));
                tournament_size = 2;
            end
            
            MatingPool = TournamentSelection(tournament_size, pool_size, FrontNo, -CrowdDis);
            
            % High-dim problems: use enhanced operator with local search
            if isHighDim && mod(total_generations, 5) == 0
                Offspring = Algorithm.EnhancedOperatorGA(Problem, Population(MatingPool), total_generations, max_initial_generations);
            else
                Offspring = OperatorGA(Problem, Population(MatingPool));
            end
            
            Population = DDD5.EnvironmentalSelection([Population, Offspring], Problem.N);
            Algorithm.Archive.add(Offspring);
            
            % Evaluate quality (now includes proximity check)
            [quality_score, nd_count, obj_range, quality_details] = Algorithm.evaluateArchiveQuality(Problem);
            quality_history = [quality_history, quality_score];
            nd_count_history = [nd_count_history, nd_count];
            
            %% CRITICAL: Call NotTerminated to record results for GUI
            if mod(total_generations, 5) == 0 || total_generations == 1
                try
                    Algorithm.NotTerminated(Population);
                catch ME
                    if strcmp(ME.identifier, 'PlatEMO:Termination')
                        disp(['  ⚠ Termination condition met at generation ', num2str(total_generations)]);
                        break;
                    else
                        rethrow(ME);
                    end
                end
            end
            
            gen_time = toc(gen_start);
            
            % Display progress every 5 generations or when close to target
            if mod(total_generations, 5) == 0 || nd_count >= min_nd_solutions * 0.8
                disp(['    Gen ', sprintf('%3d', total_generations), ...
                    ' | ND: ', sprintf('%4d', nd_count), '/', num2str(min_nd_solutions), ...
                    ' | Quality: ', sprintf('%.4f', quality_score), ...
                    ' | Proximity: ', sprintf('%.4f', quality_details.proximity_factor), ...
                    ' | f2: [', sprintf('%.1f', quality_details.min_f2), ',', sprintf('%.1f', quality_details.max_f2), ']', ...
                    ' | Time: ', sprintf('%.2fs', gen_time)]);
            end
            
            %% IMPROVED: Quality check with proximity requirement
            if isHighDim || isMultimodal
                min_proximity = 0.6;
                min_forced_gen = 100;
                min_nd_adjusted = min_nd_solutions;
                disp(['    [HIGH-DIM Ultra-Quality] Thresholds: proximity>=', num2str(min_proximity), ...
                    ', min_gen=', num2str(min_forced_gen), ', nd_req=', num2str(min_nd_adjusted)]);
            else
                min_proximity = 0.3;
                min_forced_gen = 50;
                min_nd_adjusted = min_nd_solutions;
            end
            
            quality_threshold_met = (nd_count >= min_nd_adjusted) && (quality_details.proximity_factor >= min_proximity);
            min_generations_met = total_generations >= min_forced_gen;
            
            if quality_threshold_met && min_generations_met
                disp(' ');
                disp(['  ✓ Quality threshold reached after ', num2str(total_generations), ' generations!']);
                disp(['    Non-dominated solutions: ', num2str(nd_count)]);
                disp(['    Proximity factor: ', num2str(quality_details.proximity_factor, '%.4f')]);
                disp(['    f2 range: [', num2str(quality_details.min_f2, '%.3f'), ', ', num2str(quality_details.max_f2, '%.3f'), ']']);
                break;
            end
            
            % Early stopping if stagnated - ONLY if we have enough ND solutions
            min_gen_stagnation = 100;
            if isHighDim || isMultimodal
                min_gen_stagnation = 150;
            end
            if total_generations > min_gen_stagnation && length(quality_history) >= 10
                recent_improvement = mean(quality_history(end-4:end)) - mean(quality_history(end-9:end-5));
                stagnation_threshold = 0.0005;
                if isHighDim
                    stagnation_threshold = 0.0002;
                end
                % Only stop early if we have ENOUGH non-dominated solutions
                if recent_improvement < stagnation_threshold && nd_count >= min_nd_adjusted && quality_details.proximity_factor >= min_proximity
                    disp(' ');
                    disp(['  ⚠ Early stopping due to stagnation after ', num2str(total_generations), ' generations']);
                    disp(['    Non-dominated solutions: ', num2str(nd_count), ' (target: ', num2str(min_nd_adjusted), ')']);
                    disp(['    Proximity factor: ', num2str(quality_details.proximity_factor, '%.4f')]);
                    break;
                end
            end
        end
        
        % Collect final samples
        TrainingData = Algorithm.Archive.getTrainingData();
        SamplePopulation = TrainingData;
        
        % If not enough samples, supplement with current population
        if length(SamplePopulation) < sample_size
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            nd_pop = Population(FrontNo == 1);
            needed = sample_size - length(SamplePopulation);
            if length(nd_pop) > 0 && needed > 0
                n_add = min(needed, length(nd_pop));
                SamplePopulation = [SamplePopulation, nd_pop(1:n_add)];
            end
            if length(SamplePopulation) < sample_size
                needed = sample_size - length(SamplePopulation);
                n_add = min(needed, length(Population));
                SamplePopulation = [SamplePopulation, Population(1:n_add)];
            end
        end
        
        % Limit to sample_size
        if length(SamplePopulation) > sample_size
            SamplePopulation = SamplePopulation(1:sample_size);
        end
        
        disp(' ');
        disp(['  ✓ Final sample collection: ', num2str(length(SamplePopulation)), ' solutions']);
        if total_generations >= max_initial_generations
            disp(['  ⚠ Warning: Reached maximum generations (', num2str(max_initial_generations), ')']);
        end
    end

    function [quality_score, nd_count, obj_range, quality_details] = evaluateArchiveQuality(Algorithm, Problem)
        if isempty(Algorithm.Archive) || Algorithm.Archive.getSize() == 0
            quality_score = 0;
            nd_count = 0;
            obj_range = 0;
            quality_details = struct('count_factor', 0, 'diversity_factor', 0, 'proximity_factor', 0, 'min_f1', inf, 'min_f2', inf, 'max_f2', inf);
            return;
        end
        
        AllData = Algorithm.Archive.getTrainingData();
        if isempty(AllData)
            quality_score = 0;
            nd_count = 0;
            obj_range = 0;
            quality_details = struct('count_factor', 0, 'diversity_factor', 0, 'proximity_factor', 0, 'min_f1', inf, 'min_f2', inf, 'max_f2', inf);
            return;
        end
        
        objs = AllData.objs;
        nd_count = size(objs, 1);
        min_obj = min(objs, [], 1);
        max_obj = max(objs, [], 1);
        obj_range = mean(max_obj - min_obj);
        
        diversity_factor = obj_range / 2.0;
        diversity_factor = min(1, diversity_factor);
        
        max_acceptable = 2.5;
        reasonable_range = 2.0;
        in_reasonable_range = all(objs <= max_acceptable, 2);
        proximity_factor = sum(in_reasonable_range) / max(1, size(objs, 1));
        in_ideal_range = all(objs <= reasonable_range, 2);
        ideal_proximity = sum(in_ideal_range) / max(1, size(objs, 1));
        proximity_factor = max(proximity_factor, ideal_proximity);
        
        min_f1 = min(objs(:, 1));
        min_f2 = min(objs(:, 2));
        max_f2 = max(objs(:, 2));
        
        count_factor = min(1, nd_count / Algorithm.MinNDSolutions);
        quality_score = 0.4 * count_factor + 0.2 * diversity_factor + 0.4 * proximity_factor;
        quality_details = struct('count_factor', count_factor, 'diversity_factor', diversity_factor, 'proximity_factor', proximity_factor, 'min_f1', min_f1, 'min_f2', min_f2, 'max_f2', max_f2);
    end

    function isMultimodal = isMultimodalProblem(~, Problem)
        problemName = upper(class(Problem));
        multimodalNames = {'ZDT4', 'ZDT6', 'FON', 'POL', 'VNT', 'KUR'};
        isMultimodal = false;
        for i = 1:length(multimodalNames)
            if contains(problemName, multimodalNames{i})
                isMultimodal = true;
                return;
            end
        end
        % Alternative: check if problem has many local optima by sampling
        if Problem.D >= 30
            test_samples = 10;
            objs = zeros(test_samples, Problem.M);
            for i = 1:test_samples
                dec = Problem.lower + rand(1, Problem.D).*(Problem.upper - Problem.lower);
                sol = Problem.Evaluation(dec);
                objs(i, :) = sol.objs;
            end
            obj_variance = mean(var(objs, 0, 1));
            if obj_variance > 10
                isMultimodal = true;
            end
        end
    end

    function Population = EnhancedHighDimInitialization(Algorithm, Problem)
        N = Problem.N;
        D = Problem.D;
        disp(['  [EnhancedInit] Generating ', num2str(N), ' solutions for D=', num2str(D)]);
        
        Population = Problem.Initialization();
        
        % Strategy 1.5: Add random perturbations to maintain diversity
        n_diverse = floor(N * 0.2);
        diverse_idx = randperm(N, n_diverse);
        for i = 1:n_diverse
            idx = diverse_idx(i);
            original_dec = Population(idx).dec;
            perturbation = 0.3 * randn(1, D).*(Problem.upper - Problem.lower);
            new_dec = original_dec + perturbation;
            new_dec = max(min(new_dec, Problem.upper), Problem.lower);
            Population(idx) = Problem.Evaluation(new_dec);
        end
        disp(['  [EnhancedInit] Added ', num2str(n_diverse), ' diverse solutions']);
        
        % Strategy 2: Apply local search to a subset
        n_local_search = min(floor(N * 0.2), 30);
        objs_matrix = Population.objs;
        obj_sums = objs_matrix(:, 1) + objs_matrix(:, 2);
        [~, sorted_idx] = sort(obj_sums);
        candidates = sorted_idx(1:n_local_search);
        disp(['  [EnhancedInit] Applying local search to ', num2str(n_local_search), ' candidates']);
        
        for i = 1:n_local_search
            idx = candidates(i);
            original = Population(idx);
            best_dec = original.dec;
            best_obj = sum(original.objs);
            for trial = 1:15
                step_size = 0.15 / sqrt(D);
                perturbation = step_size * randn(1, D).*(Problem.upper - Problem.lower);
                new_dec = original.dec + perturbation;
                new_dec = max(min(new_dec, Problem.upper), Problem.lower);
                new_sol = Algorithm.evaluateWithDec(Problem, new_dec);
                new_obj = sum(new_sol.objs);
                if new_obj < best_obj
                    best_dec = new_dec;
                    best_obj = new_obj;
                end
            end
            if best_obj < sum(original.objs)
                Population(idx) = Algorithm.evaluateWithDec(Problem, best_dec);
            end
        end
        
        % Strategy 3: Ensure diversity by adding some random solutions
        n_random = min(floor(N * 0.1), 20);
        random_idx = randperm(N, n_random);
        for i = 1:n_random
            random_dec = Problem.lower + rand(1, D).*(Problem.upper - Problem.lower);
            Population(random_idx(i)) = Problem.Evaluation(random_dec);
        end
        disp(['  [EnhancedInit] Completed with ', num2str(n_local_search), ' local searches']);
    end

    function Offspring = EnhancedOperatorGA(Algorithm, Problem, MatingPool, current_gen, max_gen)
        Offspring = OperatorGA(Problem, MatingPool);
        n_offspring = length(Offspring);
        n_local = max(1, floor(n_offspring * 0.15));
        offspring_objs = Offspring.objs;
        objs_sum = offspring_objs(:, 1) + offspring_objs(:, 2);
        [~, sorted_idx] = sort(objs_sum);
        D = Problem.D;
        step_size = 0.05 / sqrt(D);
        for i = 1:n_local
            idx = sorted_idx(i);
            current = Offspring(idx);
            current_obj = sum(current.objs);
            for iter = 1:30
                perturbation = step_size * randn(1, D) .* (Problem.upper - Problem.lower);
                new_dec = current.dec + perturbation;
                new_dec = max(min(new_dec, Problem.upper), Problem.lower);
                new_sol = Problem.Evaluation(new_dec);
                new_obj = sum(new_sol.objs);
                if new_obj < current_obj
                    Offspring(idx) = new_sol;
                    current_obj = new_obj;
                    current = new_sol;
                end
            end
        end
    end

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
            mutant = parent + 0.1 * randn(1, Problem.D).*(Problem.upper - Problem.lower);
            mutant = max(min(mutant, Problem.upper), Problem.lower);
            Population(mutate_idx(i)) = Problem.Evaluation(mutant);
        end
    end

    function SamplePopulation = InitialSampling(Algorithm, Problem, ga_generations, sample_size)
        Population = Problem.Initialization();
        for gen = 1:ga_generations
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            CrowdDis = CrowdingDistance(Population.objs, FrontNo);
            pool_size = max(10, floor(length(Population) * 0.5));
            MatingPool = TournamentSelection(2, pool_size, FrontNo, -CrowdDis);
            Offspring = OperatorGA(Problem, Population(MatingPool));
            Population = DDD5.EnvironmentalSelection([Population, Offspring], Problem.N);
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
                step_size = 0.1 * (1 - i / ExtraNeeded);
                mutant = parent + step_size * randn(1, Problem.D).*(Problem.upper - Problem.lower);
                mutant = max(min(mutant, Problem.upper), Problem.lower);
                ExtraDec(i, :) = mutant;
            end
            ExtraPop = Problem.Evaluation(ExtraDec);
            Population = [Population, ExtraPop];
        end
        SamplePopulation = Population;
    end

    function Population = GenerateInitialSolutions(Algorithm, Problem)
        n_solutions = Problem.N;
        TargetObjs = Algorithm.Archive.getTargetObjectives(n_solutions);
        ReferenceDecs = [];
        try
            ArchiveData = Algorithm.Archive.getTrainingData();
            if ~isempty(ArchiveData) && length(ArchiveData) >= n_solutions
                [FrontNo, ~] = NDSort(ArchiveData.objs, ArchiveData.cons, length(ArchiveData));
                NDArchive = ArchiveData(FrontNo == 1);
                if length(NDArchive) >= n_solutions
                    idx = randperm(length(NDArchive), n_solutions);
                    ReferenceDecs = NDArchive(idx).decs;
                elseif length(NDArchive) > 0
                    idx = mod(0:n_solutions-1, length(NDArchive)) + 1;
                    ReferenceDecs = NDArchive(idx).decs;
                end
            end
        catch ME
            disp(['  [GenerateInitial] Could not get reference solutions: ' ME.message]);
        end
        Dec = Algorithm.DMModel.sample(TargetObjs, ReferenceDecs);
        Dec = max(min(Dec, Problem.upper), Problem.lower);
        Population = Problem.Evaluation(Dec);
    end

    function [GAOffspring, DMOffspring, GATime, DMTime] = GenerateOffspring(Algorithm, Problem, Population)
        GAStart = tic;
        [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
        CrowdDis = CrowdingDistance(Population.objs, FrontNo);
        pool_size = max(4, floor(Problem.N * Algorithm.MatingPoolRatio));
        
        % 方案 2：根据进化进度动态调整锦标赛大小
        if Algorithm.UseDynamicTournament
            total_generations = Problem.maxFE / Problem.N;
            progress = Algorithm.Generation / total_generations;
            if progress < 0.3
                tournament_size = 2;
            elseif progress < 0.7
                tournament_size = 3;
            else
                tournament_size = 4;
            end
        else
            tournament_size = 2;
        end
        
        if Algorithm.Generation <= 5
            total_generations = Problem.maxFE / Problem.N;
            progress = Algorithm.Generation / total_generations;
            mode_str = 'DYNAMIC';
            if ~Algorithm.UseDynamicTournament
                mode_str = 'FIXED';
            end
            fprintf('    [Tournament-%s] Gen %d (%.1f%%): tournament_size = %d, pool_size = %d\n', ...
                mode_str, Algorithm.Generation, progress * 100, tournament_size, pool_size);
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
                    ReferenceDecs = [];
                    try
                        ArchiveData = Algorithm.Archive.getTrainingData();
                        if ~isempty(ArchiveData)
                            [FrontNo, ~] = NDSort(ArchiveData.objs, ArchiveData.cons, length(ArchiveData));
                            NDArchive = ArchiveData(FrontNo == 1);
                            if length(NDArchive) >= dm_count
                                idx = randperm(length(NDArchive), dm_count);
                                ReferenceDecs = NDArchive(idx).decs;
                            elseif length(NDArchive) > 0
                                idx = mod(0:dm_count-1, length(NDArchive)) + 1;
                                ReferenceDecs = NDArchive(idx).decs;
                            end
                        end
                        if size(ReferenceDecs, 1) < dm_count
                            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                            NDPop = Population(FrontNo == 1);
                            needed = dm_count - size(ReferenceDecs, 1);
                            if length(NDPop) > 0 && needed > 0
                                idx = randperm(length(NDPop), min(needed, length(NDPop)));
                                if isempty(ReferenceDecs)
                                    ReferenceDecs = NDPop(idx).decs;
                                else
                                    ReferenceDecs = [ReferenceDecs; NDPop(idx).decs];
                                end
                            end
                        end
                    catch ME
                        disp(['  [GenerateOffspring] Reference init failed: ' ME.message]);
                    end
                    Dec = Algorithm.DMModel.sample(TargetObjs, ReferenceDecs);
                    Dec = max(min(Dec, Problem.upper), Problem.lower);
                    DMOffspring = Algorithm.evaluateWithDec(Problem, Dec);
                catch ME
                    % Handle sampling error silently or log
                end
            end
        end
        DMTime = toc(DMStart);
    end

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
                    TargetObjs = DDD5.DeterministicTargetSelection(AllObjs, NormObjs, n_targets, range_obj, min_obj);
                else
                    idx = randperm(size(AllObjs, 1), n_targets);
                    TargetObjs = AllObjs(idx, :);
                end
            end
        end
    end

    function UpdateDMStats(Algorithm, DMOffspring, Population)
        if isempty(DMOffspring)
            return;
        end
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

    function displayTimeStats(Algorithm)
        disp(' ');
        disp('╔════════════════════════════════════════════════════════════╗');
        disp('║                 DDD Time Statistics                        ║');
        disp('╠════════════════════════════════════════════════════════════╣');
        ts = Algorithm.TimeStats;
        disp('║  STAGE BREAKDOWN                                           ║');
        disp(['║    STAGE 1 - Initial Sampling:    ', sprintf('%8.2f', ts.Stage1_InitialSampling), 's                    ║']);
        disp(['║    STAGE 2 - DM Training:         ', sprintf('%8.2f', ts.Stage2_DMTraining), 's                    ║']);
        disp(['║    STAGE 3 - Gen Initial Sol:     ', sprintf('%8.2f', ts.Stage3_GenInitialSolutions), 's                    ║']);
        disp('║                                                            ║');
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

    function diversity = ObjectiveSpaceDiversity(Population)
        objs = Population.objs;
        n = size(objs, 1);
        if n < 2
            diversity = 0;
            return;
        end
        min_obj = min(objs, [], 1);
        max_obj = max(objs, [], 1);
        range_obj = max_obj - min_obj;
        range_obj(range_obj == 0) = 1;
        norm_objs = (objs - min_obj) ./ range_obj;
        if n > 100
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

    function diversity = DecisionSpaceDiversity(Population)
        decs = Population.dec;
        n = size(decs, 1);
        if n < 2
            diversity = 0;
            return;
        end
        min_dec = min(decs, [], 1);
        max_dec = max(decs, [], 1);
        range_dec = max_dec - min_dec;
        fprintf('    [DEBUG] decs size: %dx%d, range: [%.6f, %.6f], mean range: %.6f\n', ...
            n, size(decs, 2), min(range_dec), max(range_dec), mean(range_dec));
        range_dec(range_dec == 0) = 1;
        norm_decs = (decs - min_dec) ./ range_dec;
        norm_range = max(norm_decs, [], 1) - min(norm_decs, [], 1);
        fprintf('    [DEBUG] norm_decs range: [%.6f, %.6f], mean: %.6f\n', ...
            min(norm_range), max(norm_range), mean(norm_range));
        if n > 100
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

    function DiagnoseCrowdingSpace(Population, threshold)
        if nargin < 2
            threshold = 0.1;
        end
        obj_div = DDD5.ObjectiveSpaceDiversity(Population);
        dec_div = DDD5.DecisionSpaceDiversity(Population);
        decs = Population.dec;
        dec_range = max(decs, [], 1) - min(decs, [], 1);
        dec_std = std(decs, 0, 1);
        fprintf('\n=== Diversity Diagnosis ===\n');
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
        fprintf('    Decision Variables Stats:\n');
        fprintf('      - Number of vars: %d\n', size(decs, 2));
        fprintf('      - Range (max-min): [%.4f, %.4f], mean=%.4f\n', min(dec_range), max(dec_range), mean(dec_range));
        fprintf('      - Std dev: [%.4f, %.4f], mean=%.4f\n', min(dec_std), max(dec_std), mean(dec_std));
        fprintf('      - Unique rows: %d / %d\n', size(unique(decs, 'rows'), 1), size(decs, 1));
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
        fprintf('  =============================\n');
    end

    function TargetObjs = DeterministicTargetSelection(AllObjs, NormObjs, n_targets, range_obj, min_obj)
        n = size(AllObjs, 1);
        M = size(NormObjs, 2);
        scores = zeros(n, 1);
        for i = 1:n
            obj_vals = NormObjs(i, :);
            dist_to_origin = norm(obj_vals);
            min_obj_val = min(obj_vals);
            scores(i) = dist_to_origin + (1 - min_obj_val);
        end
        [~, sorted_idx] = sort(scores, 'descend');
        if n >= n_targets
            step = floor(n / n_targets);
            selected_idx = sorted_idx(1:step:min(n, step*n_targets));
            if length(selected_idx) < n_targets
                remaining = setdiff(sorted_idx, selected_idx);
                n_needed = n_targets - length(selected_idx);
                selected_idx = [selected_idx; remaining(1:min(n_needed, length(remaining)))'];
            end
        else
            selected_idx = sorted_idx;
        end
        fprintf('    [DM Target Selection] kmeans failed, using deterministic selection\n');
        fprintf('    [DM Target Selection] Selected %d targets from %d candidates\n', length(selected_idx), n);
        TargetObjs = AllObjs(selected_idx, :) * range_obj + min_obj';
    end

    function Sol = evaluateWithDec(Algorithm, Problem, Dec)
        if isvector(Dec) && size(Dec, 1) == 1
            Sol = Problem.Evaluation(Dec);
            if ~isfield(Sol, 'dec')
                Sol.dec = Dec;
            end
        else
            Sol = Problem.Evaluation(Dec);
            for i = 1:size(Dec, 1)
                if ~isfield(Sol(i), 'dec')
                    Sol(i).dec = Dec(i, :);
                end
            end
        end
    end
end
end