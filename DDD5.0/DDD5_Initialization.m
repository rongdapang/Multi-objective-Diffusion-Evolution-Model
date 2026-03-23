classdef DDD5_Initialization
    methods(Static)
        function SamplePopulation = QualityDrivenInitialEvolution(Algorithm, Problem, sample_size)
            if Problem.D >= 50
                Population = DDD5_Initialization.EnhancedHighDimInitialization(Problem);
            else
                Population = Problem.Initialization();
            end
            Algorithm.Archive.add(Population);
            Algorithm.Generation = 1;
            
            hv_current = DDD5_Utils.calculateHypervolume(Population, Problem.M);
            Algorithm.HVHistory = hv_current;
            
            target_nd_ratio = 0.3;
            target_nd_count = max(10, floor(Problem.N * target_nd_ratio));
            max_generations = max(50, min(500, Problem.D * 5));
            
            stagnation_threshold = 20;
            hv_improvement_threshold = 1e-4;
            
            if Algorithm.Verbose
                disp(' ');
                disp(['  Starting evolution. Target ND solutions: ', num2str(target_nd_count)]);
                disp(['  Max generations: ', num2str(max_generations)]);
                disp(['  HV improvement threshold: ', num2str(hv_improvement_threshold)]);
            end
            
            stagnation_counter = 0;
            
            while Algorithm.Generation < max_generations
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
                CrowdDis = CrowdingDistance(Population.objs, FrontNo);
                tournament_size = 2;
                MatingPool = TournamentSelection(tournament_size, max(10, floor(length(Population) * 0.5)), FrontNo, -CrowdDis);
                Offspring = OperatorGA(Problem, Population(MatingPool));
                
                Population = DDD5_Static.EnvironmentalSelection([Population, Offspring], Problem.N);
                Algorithm.Archive.add(Offspring);
                
                nd_count = sum(NDSort(Population.objs, Population.cons, length(Population)) == 1);
                
                hv_new = DDD5_Utils.calculateHypervolume(Population, Problem.M);
                hv_improvement = (hv_new - hv_current) / max(1e-10, hv_current);
                hv_current = hv_new;
                Algorithm.HVHistory = [Algorithm.HVHistory, hv_current];
                
                if Algorithm.Verbose && mod(Algorithm.Generation, 5) == 0
                    disp(['    Gen ', num2str(Algorithm.Generation), ...
                        ' | ND: ', num2str(nd_count), ...
                        ' | HV: ', num2str(hv_current, '%.6f'), ...
                        ' | Imp: ', num2str(hv_improvement, '%.4f')]);
                end
                
                if hv_improvement < hv_improvement_threshold
                    stagnation_counter = stagnation_counter + 1;
                else
                    stagnation_counter = 0;
                end
                
                if nd_count >= target_nd_count && stagnation_counter >= stagnation_threshold
                    if Algorithm.Verbose
                        disp(['  ✓ Initial evolution converged after ', num2str(Algorithm.Generation), ' generations (HV stagnated).']);
                    end
                    break;
                end
                
                Algorithm.Generation = Algorithm.Generation + 1;
            end
            
            SamplePopulation = Algorithm.Archive.getTrainingData();
            if length(SamplePopulation) > sample_size
                SamplePopulation = SamplePopulation(1:sample_size);
            end
            if length(SamplePopulation) < sample_size
                extra = sample_size - length(SamplePopulation);
                extra_dec = Problem.lower + rand(extra, Problem.D) .* (Problem.upper - Problem.lower);
                extra_pop = Problem.Evaluation(extra_dec);
                SamplePopulation = [SamplePopulation, extra_pop];
            end
        end
        
        function Population = EnhancedHighDimInitialization(Problem)
            N = Problem.N;
            D = Problem.D;
            
            lhs = lhsdesign(N, D);
            decs = Problem.lower + lhs .* (Problem.upper - Problem.lower);
            Population = Problem.Evaluation(decs);
            
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
        
        function Population = GenerateInitialSolutions(Algorithm, Problem)
            n_solutions = Problem.N;
            ArchiveObjs = Algorithm.Archive.getObjectives();
            if size(ArchiveObjs,1) >= n_solutions
                try
                    [~, C] = kmeans(ArchiveObjs, n_solutions, 'MaxIter', 100);
                    TargetObjs = C;
                catch
                    idx = randperm(size(ArchiveObjs,1), n_solutions);
                    TargetObjs = ArchiveObjs(idx,:);
                end
            else
                TargetObjs = ArchiveObjs;
                if size(TargetObjs,1) < n_solutions
                    repeat = ceil(n_solutions / size(TargetObjs,1));
                    TargetObjs = repmat(TargetObjs, repeat, 1);
                    TargetObjs = TargetObjs(1:n_solutions, :);
                end
            end
            
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
            
            Dec = Algorithm.DMModel.sample(TargetObjs, ReferenceX);
            Dec = max(min(Dec, Problem.upper), Problem.lower);
            Population = Problem.Evaluation(Dec);
        end
    end
end