classdef DDD5 < ALGORITHM
% <2026> <multi> <real> <constrained/none>
% DDD5: Dynamic Diffusion-Driven Multi-objective Optimization Algorithm (Refactored)
%
% noise_schedule --- [0.1, 0.01] --- Noise schedule for diffusion model [start, end]
% network_structure --- [256, 256] --- Hidden layer structure (unused now, kept for compatibility)
% sample_size --- 800 --- Size of sample collection for training
% dm_epochs --- 100 --- Number of epochs for training diffusion model
% dm_steps --- 100 --- Number of diffusion steps for sampling
% archive_size --- 2000 --- Maximum size of elite solution archive
% update_interval --- 10 --- Generations between model updates
% dm_ratio --- 0.4 --- Base ratio of diffusion model offspring
% use_gpu --- true --- Whether to use GPU acceleration if available
% use_dynamic_tournament --- true --- Whether to use dynamic tournament size
% use_deterministic_dm --- true --- Whether to use deterministic DM fallback
% min_nd_solutions --- 150 --- Minimum non-dominated solutions to start diffusion
% max_initial_generations --- 100 --- Maximum generations for initial phase
% verbose --- false --- Print detailed information
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
        TimeStats               % Time statistics for each phase
        UseDynamicTournament    % Whether to use dynamic tournament size
        UseDeterministicDM      % Whether to use deterministic DM fallback
        MinNDSolutions          % Minimum non-dominated solutions to start diffusion
        MaxInitialGenerations   % Maximum generations for initial phase
        Verbose                 % Print detailed information
        HVHistory               % History of hypervolume values for stagnation detection
    end

    methods
        function main(Algorithm, Problem)
            %% Parameter setting (with automatic scaling)
            [noise_schedule, ~, sample_size, dm_epochs, dm_steps, archive_size, update_interval, dm_ratio, use_gpu, ...
                use_dynamic_tournament, use_deterministic_dm, min_nd_solutions, max_initial_generations, verbose] = ...
                Algorithm.ParameterSet([0.1, 0.01], [256, 256], 800, 100, 100, 2000, 10, 0.4, true, true, true, 150, 100, false);
            
            % Adaptive scaling of parameters based on problem dimension
            if Problem.D >= 50
                sample_size = min(2000, max(800, Problem.N * 10));
                dm_epochs = min(200, max(100, floor(Problem.D / 5)));
                dm_steps = min(200, max(50, floor(Problem.D / 2)));
                archive_size = min(5000, max(2000, Problem.N * 20));
                min_nd_solutions = min(300, max(100, floor(Problem.N * 0.3)));
                max_initial_generations = min(300, max(100, floor(Problem.D * 2)));
            end
            
            % Check ablation switches
            global ABLATION_TOURNAMENT ABLATION_DM;
            if ~isempty(ABLATION_TOURNAMENT)
                use_dynamic_tournament = ABLATION_TOURNAMENT;
            end
            if ~isempty(ABLATION_DM)
                use_deterministic_dm = ABLATION_DM;
            end
            
            Algorithm.Verbose = verbose;
            Algorithm.UseDynamicTournament = use_dynamic_tournament;
            Algorithm.UseDeterministicDM = use_deterministic_dm;
            Algorithm.MinNDSolutions = min_nd_solutions;
            Algorithm.MaxInitialGenerations = max_initial_generations;
            
            %% Initialize components
            Algorithm.Archive = SolutionArchive(archive_size, Problem.M);
            Algorithm.Scheduler = AdaptiveScheduler(dm_ratio, 0.15, 0.6);
            Algorithm.Generation = 0;
            Algorithm.DMStats = struct('total', 0, 'survived', 0, 'history', []);
            Algorithm.MatingPoolRatio = 0.6;
            Algorithm.DiversityHistory = [];
            Algorithm.HVHistory = [];
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
            if ~Algorithm.HasDLToolbox && Algorithm.Verbose
                warning('DDD:NoDLToolbox', 'Deep Learning Toolbox not found. Using enhanced GA-only mode.');
                use_gpu = false;
            end
            
            %% STAGE 1: Quality-driven initial evolution (simplified)
            if Algorithm.Verbose
                disp('╔════════════════════════════════════════════════════════════╗');
                disp('║  STAGE 1: Quality-Driven Initial Evolution                 ║');
                disp('╚════════════════════════════════════════════════════════════╝');
            end
            Stage1Start = tic;
            SamplePopulation = Algorithm.QualityDrivenInitialEvolution(Problem, sample_size);
            Algorithm.TimeStats.Stage1_InitialSampling = toc(Stage1Start);
            if Algorithm.Verbose
                disp(['  ✓ Initial evolution completed after ', num2str(Algorithm.Generation), ' generations']);
                disp(['  ✓ STAGE 1 Time: ', num2str(Algorithm.TimeStats.Stage1_InitialSampling, '%.2f'), ' seconds']);
            end
            
            %% Add initial samples to archive
            ArchiveStart = tic;
            Algorithm.Archive.add(SamplePopulation);
            if Algorithm.Verbose
                disp(['  ✓ Archive initialized with ', num2str(length(SamplePopulation)), ' solutions']);
                disp(['  ✓ Archive Add Time: ', num2str(toc(ArchiveStart), '%.2f'), ' seconds']);
            end
            
            %% STAGE 2: Train diffusion model (only if quality is sufficient)
            Stage2Start = tic;
            if Algorithm.HasDLToolbox
                try
                    TrainingData = Algorithm.Archive.getTrainingData();
                    if ~isempty(TrainingData) && length(TrainingData) >= Algorithm.MinNDSolutions
                        if Algorithm.Verbose
                            disp(' ');
                            disp('╔════════════════════════════════════════════════════════════╗');
                            disp('║  STAGE 2: Training diffusion model...                      ║');
                            disp('╚════════════════════════════════════════════════════════════╝');
                            disp(['  - Training samples: ', num2str(length(TrainingData))]);
                        end
                        Algorithm.DMModel = ConditionalDiffusionModel(Problem.D, Problem.M, noise_schedule, dm_steps, use_gpu, verbose);
                        Algorithm.DMModel = Algorithm.DMModel.train(TrainingData, Problem, dm_epochs);
                        if ~Algorithm.DMModel.IsTrained
                            warning('DDD:DMTrainingFailed', 'Diffusion model training failed. Using GA-only mode.');
                            Algorithm.HasDLToolbox = false;
                            Algorithm.DMModel = [];
                        elseif Algorithm.Verbose
                            disp('  ✓ Diffusion model trained successfully');
                        end
                    else
                        warning('DDD:InsufficientQuality', ['Insufficient non-dominated solutions (' num2str(length(TrainingData)) '). Using GA-only mode.']);
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
            if Algorithm.Verbose
                disp(['  ✓ STAGE 2 Time: ', num2str(Algorithm.TimeStats.Stage2_DMTraining, '%.2f'), ' seconds']);
            end
            
            %% STAGE 3: Generate Initial Solutions
            Stage3Start = tic;
            if Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                try
                    Population = Algorithm.GenerateInitialSolutions(Problem);
                    if Algorithm.Verbose
                        disp('  ✓ Initial solutions generated using diffusion model');
                    end
                catch ME
                    warning('DDD:DMGenError', ['Error generating initial solutions with DM: ' ME.message '. Using GA initialization.']);
                    Population = Algorithm.GAInitialization(Problem);
                    if Algorithm.Verbose
                        disp('  ✓ Initial solutions generated using GA (fallback)');
                    end
                end
            else
                Population = Algorithm.GAInitialization(Problem);
                if Algorithm.Verbose
                    disp('  ✓ Initial solutions generated using GA');
                end
            end
            
            % Add to archive
            ArchiveStart = tic;
            Algorithm.Archive.add(Population);
            Algorithm.TimeStats.Stage3_GenInitialSolutions = toc(Stage3Start);
            if Algorithm.Verbose
                disp(['  ✓ STAGE 3 Time: ', num2str(Algorithm.TimeStats.Stage3_GenInitialSolutions, '%.2f'), ' seconds']);
                disp(['  ✓ Archive Add Time: ', num2str(toc(ArchiveStart), '%.2f'), ' seconds']);
            end
            
            %% STAGE 4: Main optimization loop
            if Algorithm.Verbose
                disp('STAGE 4: Main optimization loop...');
            end
            while Algorithm.NotTerminated(Population)
                Algorithm.Generation = Algorithm.Generation + 1;
                GenStart = tic;
                
                % Update mating pool ratio dynamically
                MatingPoolStart = tic;
                Algorithm.MatingPoolRatio = Algorithm.calculateMatingPoolRatio(Population, Problem);
                MatingPoolTime = toc(MatingPoolStart);
                
                % Generate offspring
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
                
                % Environmental selection
                EnvStart = tic;
                Combined = [Population, GAOffspring, DMOffspring];
                Population = DDD5.EnvironmentalSelection(Combined, Problem.N);
                EnvTime = toc(EnvStart);
                Algorithm.TimeStats.Stage4_EnvSelection = Algorithm.TimeStats.Stage4_EnvSelection + EnvTime;
                
                % Update archive
                ArchiveStart = tic;
                if Algorithm.Generation <= 10 || mod(Algorithm.Generation, 3) == 0
                    Algorithm.Archive.add([GAOffspring, DMOffspring]);
                else
                    Algorithm.Archive.addBatch([GAOffspring, DMOffspring], Algorithm.Generation);
                end
                ArchiveTime = toc(ArchiveStart);
                Algorithm.TimeStats.Stage4_ArchiveUpdate = Algorithm.TimeStats.Stage4_ArchiveUpdate + ArchiveTime;
                
                % Update diffusion model if needed
                DMUpdateStart = tic;
                if Algorithm.HasDLToolbox && Algorithm.ShouldUpdateModel(update_interval)
                    if Algorithm.Verbose
                        disp(['Updating diffusion model at generation ' num2str(Algorithm.Generation)]);
                    end
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
                
                % Display timing
                if Algorithm.Verbose && (mod(Algorithm.Generation, 5) == 0 || Algorithm.Generation <= 3)
                    disp(['    === Gen ', num2str(Algorithm.Generation), ' Time Breakdown ===']);
                    disp(['      MatingPool Calc:   ', num2str(MatingPoolTime, '%.3f'), 's']);
                    disp(['      GA Operations:     ', num2str(GATime, '%.3f'), 's (', num2str(100 * GATime / GenTotalTime, '%.1f'), '%)']);
                    disp(['      DM Operations:     ', num2str(DMTime, '%.3f'), 's (', num2str(100 * DMTime / GenTotalTime, '%.1f'), '%)']);
                    disp(['      DM Stats Update:   ', num2str(DMStatsTime, '%.3f'), 's']);
                    disp(['      Env Selection:     ', num2str(EnvTime, '%.3f'), 's (', num2str(100 * EnvTime / GenTotalTime, '%.1f'), '%)']);
                    disp(['      Archive Add:       ', num2str(ArchiveTime, '%.3f'), 's (', num2str(100 * ArchiveTime / GenTotalTime, '%.1f'), '%)']);
                    disp(['      DM Model Update:   ', num2str(DMUpdateTime, '%.3f'), 's']);
                    disp(['      >>> GEN TOTAL:     ', num2str(GenTotalTime, '%.3f'), 's <<<']);
                end
            end
            
            % Final time statistics
            Algorithm.TimeStats.TotalTime = toc(TotalStart);
            if Algorithm.Verbose
                Algorithm.displayTimeStats();
            end
        end
    end
    
    methods(Access = private)
        %% Quality-Driven Initial Evolution (simplified with HV stagnation)
        function SamplePopulation = QualityDrivenInitialEvolution(Algorithm, Problem, sample_size)
            % Use Latin Hypercube initialization for better coverage
            if Problem.D >= 50
                Population = Algorithm.EnhancedHighDimInitialization(Problem);
            else
                Population = Problem.Initialization();
            end
            Algorithm.Archive.add(Population);
            Algorithm.Generation = 1;
            
            % Evaluate initial HV
            hv_current = Algorithm.calculateHypervolume(Population, Problem.M);
            Algorithm.HVHistory = hv_current;
            
            if Algorithm.Verbose
                disp(' ');
                disp(['  Starting evolution. Target ND solutions: ', num2str(Algorithm.MinNDSolutions)]);
                disp(['  Max generations: ', num2str(Algorithm.MaxInitialGenerations)]);
            end
            
            stagnation_counter = 0;
            stagnation_threshold = 15; % generations with no HV improvement
            hv_improvement_threshold = 1e-6;
            
            while Algorithm.Generation < Algorithm.MaxInitialGenerations
                % Generate offspring using GA only
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                CrowdDis = CrowdingDistance(Population.objs, FrontNo);
                tournament_size = 2;
                MatingPool = TournamentSelection(tournament_size, max(10, floor(length(Population) * 0.5)), FrontNo, -CrowdDis);
                Offspring = OperatorGA(Problem, Population(MatingPool));
                
                % Environmental selection
                Population = DDD5.EnvironmentalSelection([Population, Offspring], Problem.N);
                Algorithm.Archive.add(Offspring);
                
                % Evaluate quality
                nd_count = sum(NDSort(Population.objs, Population.cons, length(Population)) == 1);
                
                % Calculate HV improvement
                hv_new = Algorithm.calculateHypervolume(Population, Problem.M);
                hv_improvement = (hv_new - hv_current) / max(1e-10, hv_current);
                hv_current = hv_new;
                Algorithm.HVHistory = [Algorithm.HVHistory, hv_current];
                
                if Algorithm.Verbose && mod(Algorithm.Generation, 5) == 0
                    disp(['    Gen ', num2str(Algorithm.Generation), ...
                        ' | ND: ', num2str(nd_count), ...
                        ' | HV: ', num2str(hv_current, '%.6f'), ...
                        ' | Imp: ', num2str(hv_improvement, '%.4f')]);
                end
                
                % Check stagnation
                if hv_improvement < hv_improvement_threshold
                    stagnation_counter = stagnation_counter + 1;
                else
                    stagnation_counter = 0;
                end
                
                % Termination condition: enough ND solutions and HV stagnated
                if nd_count >= Algorithm.MinNDSolutions && stagnation_counter >= stagnation_threshold
                    if Algorithm.Verbose
                        disp(['  ✓ Initial evolution converged after ', num2str(Algorithm.Generation), ' generations (HV stagnated).']);
                    end
                    break;
                end
                
                Algorithm.Generation = Algorithm.Generation + 1;
            end
            
            % Collect final samples
            SamplePopulation = Algorithm.Archive.getTrainingData();
            if length(SamplePopulation) > sample_size
                SamplePopulation = SamplePopulation(1:sample_size);
            end
            if length(SamplePopulation) < sample_size
                % Supplement with random solutions if needed
                extra = sample_size - length(SamplePopulation);
                extra_dec = Problem.lower + rand(extra, Problem.D) .* (Problem.upper - Problem.lower);
                extra_pop = Problem.Evaluation(extra_dec);
                SamplePopulation = [SamplePopulation, extra_pop];
            end
        end
        
        %% Hypervolume calculation for population
        function hv = calculateHypervolume(~, Population, M)
            if length(Population) < 2
                hv = 0;
                return;
            end
            objs = Population.objs;
            ref = max(objs, [], 1) * 1.1;
            if M == 2
                % Exact calculation for 2D
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
                % Monte Carlo for higher dimensions
                n_samples = 5000;
                samples = rand(n_samples, M) .* ref;
                dominated = 0;
                for i = 1:n_samples
                    if any(all(samples(i,:) <= objs, 2))
                        dominated = dominated + 1;
                    end
                end
                hv = prod(ref) * dominated / n_samples;
            end
        end
        
        %% Enhanced initialization for high-dimensional problems
        function Population = EnhancedHighDimInitialization(Algorithm, Problem)
            N = Problem.N;
            D = Problem.D;
            
            % Latin Hypercube Sampling
            lhs = lhsdesign(N, D);
            decs = Problem.lower + lhs .* (Problem.upper - Problem.lower);
            Population = Problem.Evaluation(decs);
            
            % Add small perturbations to 20% of solutions
            n_perturb = floor(N * 0.2);
            perturb_idx = randperm(N, n_perturb);
            for i = 1:n_perturb
                idx = perturb_idx(i);
                perturb = 0.1 * randn(1, D) .* (Problem.upper - Problem.lower);
                new_dec = Population(idx).dec + perturb;
                new_dec = max(min(new_dec, Problem.upper), Problem.lower);
                Population(idx) = Problem.Evaluation(new_dec);
            end
        end
        
        %% GA Initialization (using archive if available)
        function Population = GAInitialization(Algorithm, Problem)
            Population = Problem.Initialization();
            if ~isempty(Algorithm.Archive) && Algorithm.Archive.getSize() > 0
                TrainingData = Algorithm.Archive.getTrainingData();
                if ~isempty(TrainingData)
                    [FrontNo, ~] = NDSort(TrainingData.objs, TrainingData.cons, length(TrainingData));
                    Elite = TrainingData(FrontNo == 1);
                    n_elite = min(length(Elite), floor(Problem.N * 0.3));
                    if n_elite > 0
                        replace_idx = randperm(Problem.N, n_elite);
                        Population(replace_idx) = Elite(1:n_elite);
                    end
                end
            end
        end
        
        %% Generate Initial Solutions using diffusion model (if available)
        function Population = GenerateInitialSolutions(Algorithm, Problem)
            n_solutions = Problem.N;
            % Select target objectives from archive (use k-means or random)
            ArchiveObjs = Algorithm.Archive.getObjectives();
            if size(ArchiveObjs,1) >= n_solutions
                % Use k-means to select diverse targets
                try
                    [~, C] = kmeans(ArchiveObjs, n_solutions, 'MaxIter', 100);
                    TargetObjs = C;
                catch
                    % Fallback to random selection
                    idx = randperm(size(ArchiveObjs,1), n_solutions);
                    TargetObjs = ArchiveObjs(idx,:);
                end
            else
                TargetObjs = ArchiveObjs;
                % Repeat if needed
                if size(TargetObjs,1) < n_solutions
                    repeat = ceil(n_solutions / size(TargetObjs,1));
                    TargetObjs = repmat(TargetObjs, repeat, 1);
                    TargetObjs = TargetObjs(1:n_solutions, :);
                end
            end
            
            % Sample reference solutions from archive
            ReferenceX = [];
            if Algorithm.Archive.getSize() > 0
                TrainingData = Algorithm.Archive.getTrainingData();
                if ~isempty(TrainingData)
                    [FrontNo, ~] = NDSort(TrainingData.objs, TrainingData.cons, length(TrainingData));
                    NDArchive = TrainingData(FrontNo == 1);
                    if length(NDArchive) >= n_solutions
                        idx = randperm(length(NDArchive), n_solutions);
                        ReferenceX = NDArchive(idx).decs;
                    elseif length(NDArchive) > 0
                        idx = mod(0:n_solutions-1, length(NDArchive)) + 1;
                        ReferenceX = NDArchive(idx).decs;
                    end
                end
            end
            
            % Generate via diffusion model
            Dec = Algorithm.DMModel.sample(TargetObjs, ReferenceX);
            Dec = max(min(Dec, Problem.upper), Problem.lower);
            Population = Problem.Evaluation(Dec);
        end
        
        %% Offspring Generation
        function [GAOffspring, DMOffspring, GATime, DMTime] = GenerateOffspring(Algorithm, Problem, Population)
            GAStart = tic;
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            CrowdDis = CrowdingDistance(Population.objs, FrontNo);
            pool_size = max(4, floor(Problem.N * Algorithm.MatingPoolRatio));
            
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
            
            MatingPool = TournamentSelection(tournament_size, pool_size, FrontNo, -CrowdDis);
            GAOffspring = OperatorGA(Problem, Population(MatingPool));
            GATime = toc(GAStart);
            
            DMStart = tic;
            DMOffspring = [];
            if Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                dm_count = Algorithm.Scheduler.getDMOffspringCount(Problem.N, Algorithm.DMStats);
                if dm_count > 0
                    try
                        % Select target objectives from current population
                        TargetObjs = Algorithm.selectTargetObjectives(Population, dm_count);
                        % Reference solutions from archive
                        ReferenceX = [];
                        if Algorithm.Archive.getSize() > 0
                            TrainingData = Algorithm.Archive.getTrainingData();
                            if ~isempty(TrainingData)
                                [FrontNo, ~] = NDSort(TrainingData.objs, TrainingData.cons, length(TrainingData));
                                NDArchive = TrainingData(FrontNo == 1);
                                if length(NDArchive) >= dm_count
                                    idx = randperm(length(NDArchive), dm_count);
                                    ReferenceX = NDArchive(idx).decs;
                                elseif length(NDArchive) > 0
                                    idx = mod(0:dm_count-1, length(NDArchive)) + 1;
                                    ReferenceX = NDArchive(idx).decs;
                                end
                            end
                        end
                        Dec = Algorithm.DMModel.sample(TargetObjs, ReferenceX);
                        Dec = max(min(Dec, Problem.upper), Problem.lower);
                        DMOffspring = Problem.Evaluation(Dec);
                    catch ME
                        % Silent fallback
                    end
                end
            end
            DMTime = toc(DMStart);
        end
        
        %% Select target objectives for DM generation
        function TargetObjs = selectTargetObjectives(Algorithm, Population, n_targets)
            PopObjs = Population.objs;
            ArchiveObjs = Algorithm.Archive.getObjectives();
            if ~isempty(ArchiveObjs)
                AllObjs = [PopObjs; ArchiveObjs];
            else
                AllObjs = PopObjs;
            end
            
            if size(AllObjs,1) <= n_targets
                TargetObjs = AllObjs;
                return;
            end
            
            % Use k-means for diversity
            try
                NormObjs = (AllObjs - min(AllObjs,[],1)) ./ max(1e-10, max(AllObjs,[],1) - min(AllObjs,[],1));
                [~, C] = kmeans(NormObjs, n_targets, 'MaxIter', 100);
                % Denormalize
                min_obj = min(AllObjs,[],1);
                range_obj = max(AllObjs,[],1) - min_obj;
                TargetObjs = C .* range_obj + min_obj;
            catch
                if Algorithm.UseDeterministicDM
                    % Deterministic selection based on objective space coverage
                    NormObjs = (AllObjs - min(AllObjs,[],1)) ./ max(1e-10, max(AllObjs,[],1) - min(AllObjs,[],1));
                    scores = sum(NormObjs, 2); % simple scoring
                    [~, idx] = sort(scores);
                    step = floor(size(AllObjs,1) / n_targets);
                    selected_idx = idx(1:step:end);
                    if length(selected_idx) > n_targets
                        selected_idx = selected_idx(1:n_targets);
                    end
                    TargetObjs = AllObjs(selected_idx, :);
                else
                    idx = randperm(size(AllObjs,1), n_targets);
                    TargetObjs = AllObjs(idx, :);
                end
            end
        end
        
        %% Update DM statistics
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
        
        %% Determine if model should be updated
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
        
        %% Update diffusion model
        function UpdateDiffusionModel(Algorithm, Problem, dm_epochs)
            if isempty(Algorithm.DMModel) || ~Algorithm.DMModel.IsTrained
                return;
            end
            TrainingData = Algorithm.Archive.getTrainingData();
            if length(TrainingData) < 10
                warning('DDD:InsufficientData', 'Insufficient data for model update');
                return;
            end
            Algorithm.DMModel = Algorithm.DMModel.train(TrainingData, Problem, dm_epochs);
        end
        
        %% Dynamic mating pool ratio
        function ratio = calculateMatingPoolRatio(Algorithm, Population, Problem)
            base_ratio = 0.6;
            total_generations = Problem.maxFE / Problem.N;
            progress = Algorithm.Generation / total_generations;
            if progress < 0.3
                ratio = base_ratio * 0.7;
            elseif progress < 0.6
                ratio = base_ratio * 0.85;
            else
                ratio = base_ratio;
            end
            diversity = Algorithm.calculatePopulationDiversity(Population);
            if diversity < 0.3
                ratio = ratio * 1.2;
            elseif diversity > 0.7
                ratio = ratio * 0.85;
            end
            ratio = max(0.3, min(0.85, ratio));
        end
        
        %% Population diversity
        function diversity = calculatePopulationDiversity(Algorithm, Population)
            if length(Population) < 2
                diversity = 0;
                return;
            end
            decs = Population.dec;
            max_dec = max(decs, [], 1);
            min_dec = min(decs, [], 1);
            range = max_dec - min_dec;
            range(range == 0) = 1;
            norm_dec = (decs - min_dec) ./ range;
            mean_dec = mean(norm_dec, 1);
            distances = sqrt(sum((norm_dec - mean_dec).^2, 2));
            diversity = mean(distances) / sqrt(size(norm_dec,2));
            diversity = min(1, diversity);
        end
        
        %% Check Deep Learning Toolbox
        function hasDL = checkDeepLearningToolbox(~)
            hasDL = false;
            try
                net = feedforwardnet([10,10]);
                hasDL = true;
                clear net;
            catch
                hasDL = false;
            end
        end
        
        %% Display time statistics
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
    end
end