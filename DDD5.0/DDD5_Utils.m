classdef DDD5_Utils
    methods(Static)
        function hv = calculateHypervolume(Population, M)
            if length(Population) < 2
                hv = 0;
                return;
            end
            objs = Population.objs;
            ref = max(objs, [], 1) * 1.1;
            if M == 2
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
            diversity = DDD5_Utils.calculatePopulationDiversity(Population);
            if diversity < 0.3
                ratio = ratio * 1.2;
            elseif diversity > 0.7
                ratio = ratio * 0.85;
            end
            ratio = max(0.3, min(0.85, ratio));
        end
        
        function diversity = calculatePopulationDiversity(Population)
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
end