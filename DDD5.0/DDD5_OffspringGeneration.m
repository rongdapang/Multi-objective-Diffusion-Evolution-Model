classdef DDD5_OffspringGeneration
    methods(Static)
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
            
            if Algorithm.Scheduler.KnowledgeTransferEnabled && Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                MatingPool = DDD5_OffspringGeneration.applyKnowledgeTransfer(Algorithm, Population, MatingPool);
            end
            
            GAOffspring = OperatorGA(Problem, Population(MatingPool));
            GATime = toc(GAStart);
            
            DMStart = tic;
            DMOffspring = [];
            if Algorithm.HasDLToolbox && ~isempty(Algorithm.DMModel) && Algorithm.DMModel.IsTrained
                dm_count = Algorithm.Scheduler.getDMOffspringCount(Problem.N, Algorithm.DMStats);
                if dm_count > 0
                    try
                        TargetObjs = DDD5_OffspringGeneration.selectTargetObjectives(Algorithm, Population, dm_count);
                        
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
                    end
                end
            end
            DMTime = toc(DMStart);
        end
        
        function MatingPool = applyKnowledgeTransfer(Algorithm, Population, MatingPool)
            n_replace = floor(length(MatingPool) * 0.2);
            
            if n_replace == 0
                return;
            end
            
            n_dm_solutions = min(n_replace, 5);
            TargetObjs = Population(MatingPool(1:n_dm_solutions)).objs;
            ReferenceX = Population(MatingPool(1:n_dm_solutions)).decs;
            
            try
                Dec = Algorithm.DMModel.sample(TargetObjs, ReferenceX);
                Dec = max(min(Dec, Algorithm.DMModel.XNormalizer.max), Algorithm.DMModel.XNormalizer.min);
                
                DM_Solutions = [];
                for i = 1:size(Dec, 1)
                    sol = struct('dec', Dec(i,:), 'objs', [], 'cons', []);
                    DM_Solutions = [DM_Solutions, sol];
                end
                
                if length(DM_Solutions) > 0
                    replace_indices = MatingPool(end-n_replace+1:end);
                    MatingPool(replace_indices) = [];
                    MatingPool = [MatingPool, 1:length(DM_Solutions)];
                end
            catch
            end
        end
        
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
            
            try
                NormObjs = (AllObjs - min(AllObjs,[],1)) ./ max(1e-10, max(AllObjs,[],1) - min(AllObjs,[],1));
                [~, C] = kmeans(NormObjs, n_targets, 'MaxIter', 100);
                
                min_obj = min(AllObjs,[],1);
                range_obj = max(AllObjs,[],1) - min_obj;
                TargetObjs = C .* range_obj + min_obj;
            catch
                if Algorithm.UseDeterministicDM
                    NormObjs = (AllObjs - min(AllObjs,[],1)) ./ max(1e-10, max(AllObjs,[],1) - min(AllObjs,[],1));
                    scores = sum(NormObjs, 2);
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
    end
end