classdef DDD5_ModelManagement
    methods(Static)
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
            Algorithm.DMModel = Algorithm.DMModel.train(TrainingData, Problem, dm_epochs);
        end
    end
end