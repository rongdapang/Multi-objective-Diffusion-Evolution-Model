%% DDD Algorithm Experimental Evaluation Script
% This script runs DDD algorithm 20 times on classic multi-objective
% optimization problems and compares with traditional algorithms
%
% Author: AI Analysis System
% Date: 2026-03-03

clc;
clear;
close all;

%% Add PlatEMO path (modify according to your installation)
addpath(genpath('PlatEMO'));

%% Experimental Configuration
config.nRuns = 20;              % Number of independent runs
config.populationSize = 100;    % Population size
config.maxEvaluations = 25000;  % Maximum function evaluations

%% Test Problems
problems = {
    @ZDT1, @ZDT2, @ZDT3, ...
    @DTLZ2, @DTLZ7, ...
    @WFG4
};

problemNames = {'ZDT1', 'ZDT2', 'ZDT3', 'DTLZ2', 'DTLZ7', 'WFG4'};

%% Algorithms to Compare
algorithms = {
    @DDD, ...
    @NSGAII, ...
    @MOEAD, ...
    @SPEA2, ...
    @RVEA
};

algorithmNames = {'DDD', 'NSGA-II', 'MOEA/D', 'SPEA2', 'RVEA'};

%% Results Storage
results = struct();
for i = 1:length(problems)
    for j = 1:length(algorithms)
        results.(problemNames{i}).(algorithmNames{j}) = struct(...
            'IGD', zeros(config.nRuns, 1), ...
            'HV', zeros(config.nRuns, 1), ...
            'GD', zeros(config.nRuns, 1), ...
            'SP', zeros(config.nRuns, 1), ...
            'Runtime', zeros(config.nRuns, 1) ...
        );
    end
end

%% Run Experiments
disp('=================================================');
disp('    DDD Algorithm Experimental Evaluation');
disp('=================================================');
disp(['Number of runs: ', num2str(config.nRuns)]);
disp(['Population size: ', num2str(config.populationSize)]);
disp(['Max evaluations: ', num2str(config.maxEvaluations)]);
disp(' ');

for p = 1:length(problems)
    problemName = problemNames{p};
    disp(['Running experiments on ', problemName, '...']);
    
    for a = 1:length(algorithms)
        algName = algorithmNames{a};
        disp(['  Algorithm: ', algName]);
        
        for run = 1:config.nRuns
            fprintf('    Run %2d/%2d: ', run, config.nRuns);
            
            try
                % Run algorithm
                tic;
                [result] = platemo('algorithm', algorithms{a}, ...
                                   'problem', problems{p}, ...
                                   'N', config.populationSize, ...
                                   'maxFE', config.maxEvaluations, ...
                                   'run', run, ...
                                   'save', 0);
                runtime = toc;
                
                % Calculate metrics
                % Get true Pareto front
                truePF = problems{p}().optimum;
                
                % Get algorithm result
                obtainedPF = result{end}.objs;
                
                % Calculate IGD
                igdValue = IGD(obtainedPF, truePF);
                
                % Calculate HV
                refPoint = max(truePF) * 1.1;
                hvValue = HV(obtainedPF, refPoint);
                
                % Calculate GD
                gdValue = GD(obtainedPF, truePF);
                
                % Calculate SP
                spValue = SP(obtainedPF);
                
                % Store results
                results.(problemName).(algName).IGD(run) = igdValue;
                results.(problemName).(algName).HV(run) = hvValue;
                results.(problemName).(algName).GD(run) = gdValue;
                results.(problemName).(algName).SP(run) = spValue;
                results.(problemName).(algName).Runtime(run) = runtime;
                
                fprintf('IGD=%.4f, HV=%.4f, Time=%.2fs\n', igdValue, hvValue, runtime);
                
            catch ME
                warning('Run %d failed: %s', run, ME.message);
                results.(problemName).(algName).IGD(run) = NaN;
                results.(problemName).(algName).HV(run) = NaN;
                results.(problemName).(algName).GD(run) = NaN;
                results.(problemName).(algName).SP(run) = NaN;
                results.(problemName).(algName).Runtime(run) = NaN;
            end
        end
        disp(' ');
    end
end

%% Statistical Analysis
disp('=================================================');
disp('           Statistical Analysis Results');
disp('=================================================');

% Create summary tables
summaryTable = [];

for p = 1:length(problems)
    problemName = problemNames{p};
    disp(['\nProblem: ', problemName]);
    disp('----------------------------------------');
    
    fprintf('%-12s %-12s %-12s %-12s %-12s\n', 'Algorithm', 'IGD(mean)', 'IGD(std)', 'HV(mean)', 'HV(std)');
    disp('----------------------------------------');
    
    for a = 1:length(algorithms)
        algName = algorithmNames{a};
        
        igdData = results.(problemName).(algName).IGD;
        hvData = results.(problemName).(algName).HV;
        
        % Remove NaN values
        igdData = igdData(~isnan(igdData));
        hvData = hvData(~isnan(hvData));
        
        if ~isempty(igdData)
            igdMean = mean(igdData);
            igdStd = std(igdData);
            hvMean = mean(hvData);
            hvStd = std(hvData);
            
            fprintf('%-12s %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
                algName, igdMean, igdStd, hvMean, hvStd);
            
            % Store for summary
            summaryTable = [summaryTable; p, a, igdMean, igdStd, hvMean, hvStd];
        end
    end
end

%% Wilcoxon Rank-Sum Test
disp(' ');
disp('=================================================');
disp('      Wilcoxon Rank-Sum Test Results (p<0.05)');
disp('=================================================');

disp(' ');
disp('DDD vs other algorithms (+: DDD better, -: DDD worse, =: no significant difference)');
disp('--------------------------------------------------------------------------------');

for p = 1:length(problems)
    problemName = problemNames{p};
    disp(['\n', problemName, ':']);
    
    dddIGD = results.(problemName).DDD.IGD;
    dddIGD = dddIGD(~isnan(dddIGD));
    
    for a = 2:length(algorithms)  % Skip DDD itself
        algName = algorithmNames{a};
        otherIGD = results.(problemName).(algName).IGD;
        otherIGD = otherIGD(~isnan(otherIGD));
        
        if length(dddIGD) >= 5 && length(otherIGD) >= 5
            [pValue, ~] = ranksum(dddIGD, otherIGD);
            
            if pValue < 0.05
                if median(dddIGD) < median(otherIGD)
                    symbol = '+';
                else
                    symbol = '-';
                end
            else
                symbol = '=';
            end
            
            fprintf('  vs %-10s: %s (p=%.4f)\n', algName, symbol, pValue);
        end
    end
end

%% Save Results
save('DDD_Experiment_Results.mat', 'results', 'config', 'summaryTable');
disp(' ');
disp('Results saved to DDD_Experiment_Results.mat');

%% Visualization
figure('Position', [100, 100, 1200, 800]);

% Plot IGD comparison
subplot(2, 2, 1);
plotMetricComparison(results, problemNames, algorithmNames, 'IGD', 'IGD (lower is better)');

% Plot HV comparison
subplot(2, 2, 2);
plotMetricComparison(results, problemNames, algorithmNames, 'HV', 'HV (higher is better)');

% Plot runtime comparison
subplot(2, 2, 3);
plotMetricComparison(results, problemNames, algorithmNames, 'Runtime', 'Runtime (seconds)');

% Plot convergence on ZDT1
subplot(2, 2, 4);
plotConvergence(results, 'ZDT1', algorithmNames);

sgtitle('DDD Algorithm Performance Comparison');

%% Save figure
saveas(gcf, 'DDD_Performance_Comparison.png');
disp('Figure saved to DDD_Performance_Comparison.png');

%% Helper Functions
function plotMetricComparison(results, problemNames, algorithmNames, metricName, yLabel)
    nProblems = length(problemNames);
    nAlgs = length(algorithmNames);
    
    data = cell(nProblems, nAlgs);
    for p = 1:nProblems
        for a = 1:nAlgs
            metricData = results.(problemNames{p}).(algorithmNames{a}).(metricName);
            data{p, a} = metricData(~isnan(metricData));
        end
    end
    
    % Create grouped box plot
    positions = [];
    groupWidth = 0.8;
    boxWidth = groupWidth / nAlgs;
    
    for p = 1:nProblems
        for a = 1:nAlgs
            pos = p - groupWidth/2 + (a-0.5) * boxWidth;
            positions = [positions, pos];
            
            if ~isempty(data{p, a})
                boxplot(data{p, a}, 'positions', pos, 'widths', boxWidth*0.8);
                hold on;
            end
        end
    end
    
    set(gca, 'XTick', 1:nProblems, 'XTickLabel', problemNames);
    ylabel(yLabel);
    xlabel('Problem');
    
    % Add legend
    legendStrings = algorithmNames;
    legend(legendStrings, 'Location', 'best');
    
    grid on;
    hold off;
end

function plotConvergence(results, problemName, algorithmNames)
    % Plot mean convergence curve (if data available)
    colors = lines(length(algorithmNames));
    
    for a = 1:length(algorithmNames)
        algName = algorithmNames{a};
        igdData = results.(problemName).(algName).IGD;
        igdData = igdData(~isnan(igdData));
        
        if ~isempty(igdData)
            meanIGD = mean(igdData);
            plot(a, meanIGD, 'o-', 'Color', colors(a, :), 'LineWidth', 2, 'MarkerSize', 10);
            hold on;
        end
    end
    
    set(gca, 'XTick', 1:length(algorithmNames), 'XTickLabel', algorithmNames);
    ylabel('Mean IGD');
    title(['Final IGD on ', problemName]);
    grid on;
    hold off;
end
