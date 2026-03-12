classdef DDD < ALGORITHM
% <2026> <multi> <real> <constrained/none>
% DDD: Dynamic Diffusion-Driven Multi-objective Optimization Algorithm
%
% This algorithm combines genetic algorithms with conditional diffusion models
% through online model adaptation. The diffusion model is continuously updated
% with elite solutions discovered during evolution.
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
        Archive         % SolutionArchive instance
        DMModel         % ConditionalDiffusionModel instance
        Scheduler       % AdaptiveScheduler instance
        Generation      % Current generation counter
        DMStats         % Statistics for DM offspring quality
        HasDLToolbox    % Flag indicating if Deep Learning Toolbox is available
    end

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [noise_schedule, network_structure, ga_generations, sample_size, dm_epochs, dm_steps, archive_size, update_interval, dm_ratio, use_gpu] = Algorithm.ParameterSet([0.1, 0.01], [256, 512, 512, 256], 20, 500, 100, 50, 1000, 10, 0.4, true);
            
            %% Initialize components
            Algorithm.Archive = SolutionArchive(archive_size, Problem.M);
            Algorithm.Scheduler = AdaptiveScheduler(dm_ratio, 15, 50);
            Algorithm.Generation = 0;
            Algorithm.DMStats = struct('total', 0, 'survived', 0, 'history', []);
            
            %% Check for Deep Learning Toolbox
            Algorithm.HasDLToolbox = Algorithm.checkDeepLearningToolbox();
            
            if ~Algorithm.HasDLToolbox
                warning('DDD:NoDLToolbox', 'Deep Learning Toolbox not found. Using enhanced GA-only mode.');
                use_gpu = false;
            end
            
            %% STAGE 1: High-quality initial sampling with GA
            disp('STAGE 1: Initial sampling with GA...');
            SamplePopulation = Algorithm.InitialSampling(Problem, ga_generations, sample_size);
            
            %% Add initial samples to archive
            Algorithm.Archive.add(SamplePopulation);
            
            %% STAGE 2: Train diffusion model
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
            
            %% STAGE 3: Generate initial solutions
            disp('STAGE 3: Generating initial solutions...');
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
            Algorithm.Archive.add(Population);
            
            %% STAGE 4: Main optimization loop
            disp('STAGE 4: Main optimization loop...');
            while Algorithm.NotTerminated(Population)
                Algorithm.Generation = Algorithm.Generation + 1;
                
                %% Generate offspring with both GA and DM
                [GAOffspring, DMOffspring] = Algorithm.GenerateOffspring(Problem, Population);
                
                %% Update DM statistics
                if ~isempty(DMOffspring)
                    Algorithm.UpdateDMStats(DMOffspring, Population);
                end
                
                %% Environmental selection using NSGA-II mechanism
                Combined = [Population, GAOffspring, DMOffspring];
                Population = DDD.EnvironmentalSelection(Combined, Problem.N);
                
                %% Update archive with new solutions
                Algorithm.Archive.add([GAOffspring, DMOffspring]);
                
                %% Update diffusion model if needed
                if Algorithm.HasDLToolbox && Algorithm.ShouldUpdateModel(update_interval)
                    disp(['Updating diffusion model at generation ' num2str(Algorithm.Generation)]);
                    try
                        Algorithm.UpdateDiffusionModel(Problem, dm_epochs);
                    catch ME
                        warning('DDD:DMUpdateError', ['Error updating diffusion model: ' ME.message]);
                    end
                end
            end
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
                MatingPool = TournamentSelection(2, length(Population), FrontNo, -CrowdDis);
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
        
        %% Generate Offspring
        function [GAOffspring, DMOffspring] = GenerateOffspring(Algorithm, Problem, Population)
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            CrowdDis = CrowdingDistance(Population.objs, FrontNo);
            MatingPool = TournamentSelection(2, Problem.N, FrontNo, -CrowdDis);
            GAOffspring = OperatorGA(Problem, Population(MatingPool));
            
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
                    idx = randperm(size(AllObjs, 1), n_targets);
                    TargetObjs = AllObjs(idx, :);
                end
            end
        end
        
        %% Update DM Statistics
        function UpdateDMStats(Algorithm, DMOffspring, Population)
            if isempty(DMOffspring)
                return;
            end
            
            n_survived = 0;
            for i = 1:length(DMOffspring)
                for j = 1:length(Population)
                    if isequal(DMOffspring(i).dec, Population(j).dec)
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
    end
end
