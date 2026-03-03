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
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties (Access = private)
        Archive         % SolutionArchive instance
        DMModel         % ConditionalDiffusionModel instance
        Scheduler       % AdaptiveScheduler instance
        Generation      % Current generation counter
        DMStats         % Statistics for DM offspring quality
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
            
            %% Check for Deep Learning Toolbox
            hasDLToolbox = license('test', 'Deep_Learning_Toolbox');
            if ~hasDLToolbox
                warning('Deep Learning Toolbox not found. Using fallback GA-only mode.');
                use_gpu = false;
            end
            
            %% STAGE 1: High-quality initial sampling with GA
            disp('STAGE 1: Initial sampling with GA...');
            SamplePopulation = Algorithm.InitialSampling(Problem, ga_generations, sample_size);
            
            %% Add initial samples to archive
            Algorithm.Archive.add(SamplePopulation);
            
            %% STAGE 2: Train diffusion model
            if hasDLToolbox
                disp('STAGE 2: Training diffusion model...');
                Algorithm.DMModel = ConditionalDiffusionModel(Problem.D, Problem.M, ...
                    network_structure, noise_schedule, dm_steps, use_gpu);
                Algorithm.DMModel = Algorithm.DMModel.train(SamplePopulation, Problem, dm_epochs);
            else
                Algorithm.DMModel = [];
            end
            
            %% STAGE 3: Generate initial solutions using diffusion model
            disp('STAGE 3: Generating initial solutions...');
            if ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                Population = Algorithm.GenerateInitialSolutions(Problem);
            else
                Population = Problem.Initialization();
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
                Population = EnvironmentalSelection(Combined, Problem.N);
                
                %% Update archive with new solutions
                Algorithm.Archive.add([GAOffspring, DMOffspring]);
                
                %% Update diffusion model if needed
                if Algorithm.ShouldUpdateModel(update_interval)
                    disp(['Updating diffusion model at generation ', num2str(Algorithm.Generation)]);
                    Algorithm.UpdateDiffusionModel(Problem, dm_epochs);
                end
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
                Population = EnvironmentalSelection([Population, Offspring], Problem.N);
            end
            
            % Fill to sample_size with guided mutation if needed
            CurrentSize = length(Population);
            if CurrentSize < sample_size
                ExtraNeeded = sample_size - CurrentSize;
                ExtraDec = zeros(ExtraNeeded, Problem.D);
                
                % Get non-dominated solutions for guidance
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                ElitePop = Population(FrontNo == 1);
                
                for i = 1:ExtraNeeded
                    % Select random elite parent
                    parent = ElitePop(randi(length(ElitePop))).dec;
                    % Guided mutation with adaptive step size
                    step_size = 0.1 * (1 - gen/ga_generations);
                    mutant = parent + step_size * randn(1, Problem.D) .* (Problem.upper - Problem.lower);
                    mutant = max(min(mutant, Problem.upper), Problem.lower);
                    ExtraDec(i, :) = mutant;
                end
                
                ExtraPop = Problem.Evaluation(ExtraDec);
                Population = [Population, ExtraPop];
            end
            
            % Select diverse samples using crowding distance
            if length(Population) > sample_size
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                CrowdDis = CrowdingDistance(Population.objs, FrontNo);
                [~, idx] = sort(-CrowdDis);
                SamplePopulation = Population(idx(1:sample_size));
            else
                SamplePopulation = Population;
            end
        end
        
        %% ==================== Initial Solution Generation ====================
        function Population = GenerateInitialSolutions(Algorithm, Problem)
            if isempty(Algorithm.DMModel) || ~Algorithm.DMModel.IsTrained
                Population = Problem.Initialization();
                return;
            end
            
            nElite = round(Problem.N * 0.3);
            nDiverse = Problem.N - nElite;
            
            % Get reference points from archive
            refPoints = Algorithm.Archive.getReferencePoints(nElite);
            EliteDec = Algorithm.DMModel.generate(nElite, refPoints, 'elite');
            
            % Generate diverse samples
            DiverseDec = Algorithm.DMModel.generate(nDiverse, [], 'diverse');
            
            AllDec = [EliteDec; DiverseDec];
            AllDec = max(min(AllDec, Problem.upper), Problem.lower);
            Population = Problem.Evaluation(AllDec);
        end
        
        %% ==================== Offspring Generation ====================
        function [GAOffspring, DMOffspring] = GenerateOffspring(Algorithm, Problem, Population)
            N = Problem.N;
            
            % Get adaptive DM ratio
            metrics = Algorithm.CollectMetrics();
            dmRatio = Algorithm.Scheduler.getDMRatio(Algorithm.Generation, metrics);
            
            nDM = round(N * dmRatio);
            nGA = N - nDM;
            
            % GA offspring
            [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
            CrowdDis = CrowdingDistance(Population.objs, FrontNo);
            MatingPool = TournamentSelection(2, nGA * 2, FrontNo, -CrowdDis);
            GAOffspring = OperatorGA(Problem, Population(MatingPool));
            if length(GAOffspring) > nGA
                GAOffspring = GAOffspring(1:nGA);
            end
            
            % DM offspring
            DMOffspring = [];
            if ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained && nDM > 0
                % Select diverse parents based on crowding distance
                [~, idx] = sort(-CrowdDis);
                selectedIdx = idx(1:min(nDM, length(idx)));
                if length(selectedIdx) < nDM
                    selectedIdx = [selectedIdx, randi(length(Population), 1, nDM - length(selectedIdx))];
                end
                
                % Use objectives as conditioning targets
                F_target = reshape([Population(selectedIdx).obj], Problem.M, [])';
                DMDec = Algorithm.ArchiveGuidedSampling(F_target);
                DMDec = max(min(DMDec, Problem.upper), Problem.lower);
                DMOffspring = Problem.Evaluation(DMDec);
            elseif nDM > 0
                % Fallback to random sampling
                DMDec = Problem.lower + rand(nDM, Problem.D) .* (Problem.upper - Problem.lower);
                DMOffspring = Problem.Evaluation(DMDec);
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
                % Check if this solution is non-dominated in combined population
                isSurvived = any(Population == DMOffspring(i));
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
end
