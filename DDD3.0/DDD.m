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
            [noise_schedule, network_structure, ga_generations, sample_size, ...
             dm_epochs, dm_steps, archive_size, update_interval, dm_ratio, use_gpu] = ...
                Algorithm.ParameterSet([0.1, 0.01], [256, 512, 512, 256], 20, 500, ...
                                       100, 50, 1000, 10, 0.4, true);
            
            %% Initialize components
            Algorithm.Archive = SolutionArchive(archive_size, Problem.M);
            Algorithm.Scheduler = AdaptiveScheduler(dm_ratio, 15, 50);
            Algorithm.Generation = 0;
            Algorithm.DMStats = struct('total', 0, 'survived', 0, 'history', []);
            
            %% Check for Deep Learning Toolbox with multiple methods
            Algorithm.HasDLToolbox = Algorithm.checkDeepLearningToolbox();
            
            if ~Algorithm.HasDLToolbox
                warning('DDD:NoDLToolbox', ...
                    ['Deep Learning Toolbox not found. Using enhanced GA-only mode.\n' ...
                     'To enable diffusion model features, please install Deep Learning Toolbox.']);
                use_gpu = false;
            else
                % Additional check for required functions
                if ~Algorithm.checkRequiredFunctions()
                    warning('DDD:MissingFunctions', ...
                        ['Some required functions are missing. Using enhanced GA-only mode.']);
                    Algorithm.HasDLToolbox = false;
                    use_gpu = false;
                end
            end
            
            %% STAGE 1: High-quality initial sampling with GA
            disp('STAGE 1: Initial sampling with GA...');
            SamplePopulation = Algorithm.InitialSampling(Problem, ga_generations, sample_size);
            
            %% Add initial samples to archive
            Algorithm.Archive.add(SamplePopulation);
            
            %% STAGE 2: Train diffusion model (if toolbox available)
            if Algorithm.HasDLToolbox
                try
                    disp('STAGE 2: Training diffusion model...');
                    Algorithm.DMModel = ConditionalDiffusionModel(Problem.D, Problem.M, ...
                        network_structure, noise_schedule, dm_steps, use_gpu);
                    Algorithm.DMModel = Algorithm.DMModel.train(SamplePopulation, Problem, dm_epochs);
                    
                    if ~Algorithm.DMModel.IsTrained
                        warning('DDD:DMTrainingFailed', ...
                            'Diffusion model training failed. Falling back to GA-only mode.');
                        Algorithm.HasDLToolbox = false;
                        Algorithm.DMModel = [];
                    end
                catch ME
                    warning('DDD:DMInitError', ...
                        ['Error initializing diffusion model: ' ME.message '\nFalling back to GA-only mode.']);
                    Algorithm.HasDLToolbox = false;
                    Algorithm.DMModel = [];
                end
            else
                Algorithm.DMModel = [];
            end
            
            %% STAGE 3: Generate initial solutions using diffusion model or GA
            disp('STAGE 3: Generating initial solutions...');
            if Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                try
                    Population = Algorithm.GenerateInitialSolutions(Problem);
                catch ME
                    warning('DDD:DMGenError', ...
                        ['Error generating initial solutions with DM: ' ME.message '\nUsing GA initialization.']);
                    Population = Problem.Initialization();
                end
            else
                % Enhanced GA-based initialization for fallback mode
                Population = Algorithm.EnhancedGAInitialization(Problem, SamplePopulation);
            end
            
            %% Add to archive
            Algorithm.Archive.add(Population);
            
            %% STAGE 4: Main optimization loop
            disp('STAGE 4: Main optimization loop...');
            while Algorithm.NotTerminated(Population)
                Algorithm.Generation = Algorithm.Generation + 1;
                
                %% Generate offspring with both GA and DM (if available)
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
                
                %% Update diffusion model if needed (and available)
                if Algorithm.HasDLToolbox && Algorithm.ShouldUpdateModel(update_interval)
                    try
                        disp(['Updating diffusion model at generation ', num2str(Algorithm.Generation)]);
                        Algorithm.UpdateDiffusionModel(Problem, dm_epochs);
                    catch ME
                        warning('DDD:DMUpdateError', ...
                            ['Error updating diffusion model: ' ME.message '\nContinuing with GA.']);
                    end
                end
            end
        end
        
        %% Check if Deep Learning Toolbox is available
        function hasDL = checkDeepLearningToolbox(~)
            hasDL = false;
            
            % Method 1: Check license
            try
                hasDL = license('test', 'Deep_Learning_Toolbox');
            catch
                hasDL = false;
            end
            
            % Method 2: Check for required functions
            if hasDL
                required_funcs = {'feedforwardnet', 'train', 'sim', 'mapminmax'};
                for i = 1:length(required_funcs)
                    if ~exist(required_funcs{i}, 'file')
                        hasDL = false;
                        return;
                    end
                end
            end
            
            % Method 3: Try to create a simple network
            if hasDL
                try
                    test_net = feedforwardnet(1);
                    clear test_net;
                catch
                    hasDL = false;
                end
            end
        end
        
        %% Check for required functions
        function hasFuncs = checkRequiredFunctions(~)
            hasFuncs = true;
            required_funcs = {'feedforwardnet', 'train', 'sim', 'mapminmax', 'init'};
            for i = 1:length(required_funcs)
                if ~exist(required_funcs{i}, 'file')
                    hasFuncs = false;
                    return;
                end
            end
        end
        
        %% Enhanced GA-based initialization for fallback mode
        function Population = EnhancedGAInitialization(Algorithm, Problem, SamplePopulation)
            % Use sample population as base and add diversity
            Population = Problem.Initialization();
            
            % Inject good solutions from sample population
            if ~isempty(SamplePopulation)
                [FrontNo, ~] = NDSort(SamplePopulation.objs, SamplePopulation.cons, length(SamplePopulation));
                ElitePop = SamplePopulation(FrontNo == 1);
                
                % Replace some random solutions with elite solutions
                n_elite = min(length(ElitePop), floor(Problem.N * 0.3));
                if n_elite > 0
                    replace_idx = randperm(Problem.N, n_elite);
                    Population(replace_idx) = ElitePop(1:n_elite);
                end
            end
            
            % Add guided mutation for diversity
            n_mutate = floor(Problem.N * 0.2);
            mutate_idx = randperm(Problem.N, n_mutate);
            
            for i = 1:n_mutate
                parent = Population(mutate_idx(i)).dec;
                mutant = parent + 0.1 * randn(1, Problem.D) .* (Problem.upper - Problem.lower);
                mutant = max(min(mutant, Problem.upper), Problem.lower);
                Population(mutate_idx(i)) = Problem.Evaluation(mutant);
            end
        end
        
        %% ==================== Initial Sampling ====================
        function SamplePopulation = InitialSampling(Algorithm, Problem, ga_generations, sample_size)
            % Initialize population
            Population = Problem.Initialization();
            
            % Run GA for initial sampling
            for gen = 1:ga_generations
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                CrowdDis = CrowdingDistance(Population.objs, FrontNo);
                MatingPool = TournamentSelection(2, length(Population), FrontNo, -CrowdDis);
                Offspring = OperatorGA(Problem, Population(MatingPool));
                Population = DDD.EnvironmentalSelection([Population, Offspring], Problem.N);
            end
            
            % Fill to sample_size with guided mutation if needed
            CurrentSize = length(Population);
            if CurrentSize < sample_size
                ExtraNeeded = sample_size - CurrentSize;
                ExtraDec = zeros(ExtraNeeded, Problem.D);
                
                % Get non-dominated solutions for guidance
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                ElitePop = Population(FrontNo == 1);
                
                % Fallback to all population if no non-dominated solutions
                if isempty(Eli  selectedIdx = [selectedIdx, randi(length(Population), 1, nDM - length(selectedIdx))];
                    end
                    
                    % Use objectives as conditioning targets
                    F_target = reshape([Population(selectedIdx).obj], Problem.M, [])';
                    DMDec = Algorithm.ArchiveGuidedSampling(F_target);
                    if ~isempty(DMDec)
                        DMDec = max(min(DMDec, Problem.upper), Problem.lower);
                        DMOffspring = Problem.Evaluation(DMDec);
                    end
                end
                
                % Fallback to random sampling if DM failed or not available
                if isempty(DMOffspring)
                    DMDec = Problem.lower + rand(nDM, Problem.D) .* (Problem.upper - Problem.lower);
                    DMOffspring = Problem.Evaluation(DMDec);
                end
            end
        end
        
        %% ==================== Archive-Guided Sampling ====================
        function X = ArchiveGuidedSampling(Algorithm, F_target)
            if isempty(Algorithm.DMModel) || ~Algorithm.DMModel.IsTrained
                X = [];
                return;
            end
            
            nSamples = size(F_target, 1);
            
            % Normalize target objectives
            F_norm = Algorithm.DMModel.normalizeObjectives(F_target);
            F_norm = max(0, min(1, F_norm));
            
            % Generate samples conditioned on target objectives
            X = Algorithm.DMModel.generate(nSamples, F_norm, 'guided');
        end
        
        %% ==================== Model Update ====================
        function shouldUpdate = ShouldUpdateModel(Algorithm, interval)
            shouldUpdate = ~isempty(Algorithm.DMModel) && ...
                          Algorithm.DMModel.IsTrained && ...
                          mod(Algorithm.Generation, interval) == 0 && ...
                          Algorithm.Generation > 20;
        end
        
        function UpdateDiffusionModel(Algorithm, Problem, epochs)
            if isempty(Algorithm.DMModel) || ~Algorithm.DMModel.IsTrained
                return;
            end
            
            % Get training data from archive
            trainingData = Algorithm.Archive.getTrainingData(500);
            
            % Fine-tune model with new data
            Algorithm.DMModel = Algorithm.DMModel.fineTune(trainingData, Problem, round(epochs * 0.3));
        end
        
        %% ==================== Statistics Collection ====================
        function metrics = CollectMetrics(Algorithm)
            metrics = struct();
            metrics.generation = Algorithm.Generation;
            
            if Algorithm.DMStats.total > 0
                metrics.dmSurvivalRate = Algorithm.DMStats.survived / Algorithm.DMStats.total;
            else
                metrics.dmSurvivalRate = 0.3;
            end
            
            if length(Algorithm.DMStats.history) >= 5
                recent = Algorithm.DMStats.history(end-4:end);
                metrics.stagnationGenerations = sum(diff(recent) < 0.01);
            else
                metrics.stagnationGenerations = 0;
            end
            
            metrics.archiveSize = Algorithm.Archive.Size;
        end
        
        function UpdateDMStats(Algorithm, DMOffspring, Population)
            Algorithm.DMStats.total = Algorithm.DMStats.total + length(DMOffspring);
            
            % Count how many DM offspring survived to next generation
            for i = 1:length(DMOffspring)
                % Check if this solution is in the population (survived)
                isSurvived = false;
                for j = 1:length(Population)
                    if isequal(DMOffspring(i).dec, Population(j).dec)
                        isSurvived = true;
                        break;
                    end
                end
                if isSurvived
                    Algorithm.DMStats.survived = Algorithm.DMStats.survived + 1;
                end
            end
            
            % Update history
            if Algorithm.DMStats.total > 0
                Algorithm.DMStats.history = [Algorithm.DMStats.history, ...
                    Algorithm.DMStats.survived / Algorithm.DMStats.total];
            end
        end
    end
    
    %% ==================== Static Methods ====================
    methods (Static)
        function [Population, FrontNo, CrowdDis] = EnvironmentalSelection(Population, N)
            % Environmental selection based on NSGA-II
            [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
            Next = FrontNo < MaxFNo;
            
            % Select remaining individuals based on crowding distance
            Last = find(FrontNo == MaxFNo);
            if sum(Next) + length(Last) > N
                % Calculate crowding distance for the last front
                CrowdDis = CrowdingDistance(Population(Last).objs, ones(1, length(Last)));
                [~, idx] = sort(-CrowdDis);
                selected = Last(idx(1:N-sum(Next)));
                Next(selected) = true;
            else
                Next(Last) = true;
            end
            
            Population = Population(Next);
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            CrowdDis = CrowdingDistance(Population.objs, FrontNo);
        end
    end
end
ing);
            Algorithm.DMStats.survived = Algorithm.DMStats.survived + n_survived;
            
            % Calculate survival rate
            survival_rate = n_survived / length(DMOffspring);
            Algorithm.DMStats.history = [Algorithm.DMStats.history, survival_rate];
            
            % Keep only recent history
            if length(Algorithm.DMStats.history) > 50
                Algorithm.DMStats.history = Algorithm.DMStats.history(end-49:end);
            end
        end
        
        %% ==================== Should Update Model ====================
        function shouldUpdate = ShouldUpdateModel(Algorithm, update_interval)
            shouldUpdate = false;
            
            % Update based on interval
            if mod(Algorithm.Generation, update_interval) == 0
                shouldUpdate = true;
                return;
            end
            
            % Update if DM performance is poor
            if ~isempty(Algorithm.DMStats.history) && length(Algorithm.DMStats.history) >= 5
                recent_performance = mean(Algorithm.DMStats.history(end-4:end));
                if recent_performance < 0.1  % Less than 10% survival rate
                    shouldUpdate = true;
                end
            end
        end
        
        %% ==================== Update Diffusion Model ====================
        function UpdateDiffusionModel(Algorithm, Problem, dm_epochs)
            if isempty(Algorithm.DMModel) || ~Algorithm.DMModel.IsTrained
                return;
            end
            
            % Get training data from archive
            TrainingData = Algorithm.Archive.getTrainingData();
            
            if length(TrainingData) < 10
                warning('DDD:InsufficientData', 'Insufficient data for model update');
                return;
            end
            
            % Retrain model
            try
                Algorithm.DMModel = Algorithm.DMModel.train(TrainingData, Problem, dm_epochs);
            catch ME
                warning('DDD:DMRetrainError', ...
                    ['Error retraining diffusion model: ' ME.message]);
            end
        end
    end
    
    methods(Static)
        %% ==================== Environmental Selection ====================
        function Population = EnvironmentalSelection(Population, N)
            % Non-dominated sorting
            [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
            Next = FrontNo < MaxFNo;
            
            % Select remaining solutions based on crowding distance
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
