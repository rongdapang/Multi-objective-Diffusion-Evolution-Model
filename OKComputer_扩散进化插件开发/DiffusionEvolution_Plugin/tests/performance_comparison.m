%% Performance Comparison Script
% This script compares the performance of DiffusionEvolution with other
% algorithms (NSGA-II, NSGA-III) on various test problems.
%
% Results are saved to performance_results.mat for analysis.

%% Setup
clear all;
clc;

% Add paths
% addpath('path/to/PlatEMO');
% addpath('path/to/DiffusionEvolution');

% Test configuration
numRuns = 5;  % Number of independent runs
problems = {'ZDT1', 'ZDT2', 'ZDT3', 'DTLZ2', 'CONSTR'};
algorithms = {'DiffusionEvolution', 'NSGAII', 'NSGAIII'};

% Algorithm configurations
DE_config = {100, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness'};
NSGAII_config = {};  % Default configuration
NSGAIII_config = {}; % Default configuration

% Results storage
results = struct();
results.problems = problems;
results.algorithms = algorithms;
results.metrics = {'HV', 'IGD', 'GD', 'Feasible_rate', 'runtime'};

%% Performance comparison loop
fprintf('=== Performance Comparison ===\n');
fprintf('Problems: %s\n', strjoin(problems, ', '));
fprintf('Algorithms: %s\n', strjoin(algorithms, ', '));
fprintf('Runs per problem-algorithm pair: %d\n\n', numRuns);

for probIdx = 1:length(problems)
    problemName = problems{probIdx};
    fprintf('Testing problem: %s\n', problemName);
    
    % Create problem instance
    try
        Problem = eval(problemName);
    catch
        warning('Problem %s not found, skipping...', problemName);
        continue;
    end
    
    % Get true Pareto front if available
    if isfield(Problem, 'optimum') && ~isempty(Problem.optimum)
        truePF = Problem.optimum;
    else
        truePF = [];
    end
    
    for algIdx = 1:length(algorithms)
        algorithmName = algorithms{algIdx};
        fprintf('  Algorithm: %s... ', algorithmName);
        
        % Initialize results for this combination
        results.(problemName).(algorithmName).HV = zeros(numRuns, 1);
        results.(problemName).(algorithmName).IGD = zeros(numRuns, 1);
        results.(problemName).(algorithmName).GD = zeros(numRuns, 1);
        results.(problemName).(algorithmName).FeasibleRate = zeros(numRuns, 1);
        results.(problemName).(algorithmName).Runtime = zeros(numRuns, 1);
        results.(problemName).(algorithmName).Evaluations = zeros(numRuns, 1);
        
        for run = 1:numRuns
            try
                % Create algorithm instance
                switch algorithmName
                    case 'DiffusionEvolution'
                        Algorithm = DiffusionEvolution('parameter', DE_config);
                    case 'NSGAII'
                        Algorithm = NSGAII();
                    case 'NSGAIII'
                        Algorithm = NSGAIII();
                end
                
                % Run algorithm
                Algorithm.Solve(Problem);
                
                % Get final population
                finalPop = Algorithm.result{end}{2};
                
                % Calculate metrics
                % HV (Hypervolume)
                try
                    hv_value = HV(finalPop);
                    results.(problemName).(algorithmName).HV(run) = hv_value;
                catch
                    results.(problemName).(algorithmName).HV(run) = NaN;
                end
                
                % IGD (Inverted Generational Distance)
                if ~isempty(truePF)
                    try
                        igd_value = IGD(finalPop, truePF);
                        results.(problemName).(algorithmName).IGD(run) = igd_value;
                    catch
                        results.(problemName).(algorithmName).IGD(run) = NaN;
                    end
                else
                    results.(problemName).(algorithmName).IGD(run) = NaN;
                end
                
                % GD (Generational Distance)
                if ~isempty(truePF)
                    try
                        gd_value = GD(finalPop, truePF);
                        results.(problemName).(algorithmName).GD(run) = gd_value;
                    catch
                        results.(problemName).(algorithmName).GD(run) = NaN;
                    end
                else
                    results.(problemName).(algorithmName).GD(run) = NaN;
                end
                
                % Feasible rate (for constrained problems)
                try
                    cons = [finalPop.cons];
                    if ~isempty(cons)
                        feasible = all(cons <= 0, 2);
                        feasibleRate = sum(feasible) / length(feasible);
                        results.(problemName).(algorithmName).FeasibleRate(run) = feasibleRate;
                    else
                        results.(problemName).(algorithmName).FeasibleRate(run) = 1.0;
                    end
                catch
                    results.(problemName).(algorithmName).FeasibleRate(run) = NaN;
                end
                
                % Runtime
                results.(problemName).(algorithmName).Runtime(run) = Algorithm.metric.runtime;
                
                % Evaluations
                results.(problemName).(algorithmName).Evaluations(run) = Algorithm.pro.FE;
                
            catch ME
                fprintf('Run %d failed: %s\n', run, ME.message);
                results.(problemName).(algorithmName).HV(run) = NaN;
                results.(problemName).(algorithmName).IGD(run) = NaN;
                results.(problemName).(algorithmName).GD(run) = NaN;
                results.(problemName).(algorithmName).FeasibleRate(run) = NaN;
                results.(problemName).(algorithmName).Runtime(run) = NaN;
                results.(problemName).(algorithmName).Evaluations(run) = NaN;
            end
        end
        
        fprintf('completed\n');
    end
    fprintf('\n');
end

%% Calculate statistics
fprintf('Calculating statistics...\n');
for probIdx = 1:length(problems)
    problemName = problems{probIdx};
    for algIdx = 1:length(algorithms)
        algorithmName = algorithms{algIdx};
        
        % Calculate mean and std for each metric
        for metricIdx = 1:length(results.metrics)
            metricName = results.metrics{metricIdx};
            data = results.(problemName).(algorithmName).(metricName);
            
            % Remove NaN values
            validData = data(~isnan(data));
            
            if ~isempty(validData)
                results.(problemName).(algorithmName).(['Mean_' metricName]) = mean(validData);
                results.(problemName).(algorithmName).(['Std_' metricName]) = std(validData);
            else
                results.(problemName).(algorithmName).(['Mean_' metricName]) = NaN;
                results.(problemName).(algorithmName).(['Std_' metricName]) = NaN;
            end
        end
    end
end

%% Display results summary
fprintf('\n=== Performance Summary ===\n');
fprintf('Format: Mean (Std)\n\n');

for probIdx = 1:length(problems)
    problemName = problems{probIdx};
    fprintf('Problem: %s\n', problemName);
    fprintf('%-20s', 'Algorithm');
    for metricIdx = 1:length(results.metrics)
        fprintf('%-15s', results.metrics{metricIdx});
    end
    fprintf('\n');
    
    for algIdx = 1:length(algorithms)
        algorithmName = algorithms{algIdx};
        fprintf('%-20s', algorithmName);
        
        for metricIdx = 1:length(results.metrics)
            metricName = results.metrics{metricIdx};
            meanVal = results.(problemName).(algorithmName).(['Mean_' metricName]);
            stdVal = results.(problemName).(algorithmName).(['Std_' metricName]);
            
            if isnan(meanVal)
                fprintf('%-15s', 'N/A');
            else
                fprintf('%-15s', sprintf('%.4e (%.2e)', meanVal, stdVal));
            end
        end
        fprintf('\n');
    end
    fprintf('\n');
end

%% Create comparison plots
fprintf('Creating comparison plots...\n');

% HV comparison
figure('Name', 'Hypervolume Comparison');
subplot(2, 1, 1);
boxplotData = [];
boxplotLabels = {};
for probIdx = 1:length(problems)
    problemName = problems{probIdx};
    for algIdx = 1:length(algorithms)
        algorithmName = algorithms{algIdx};
        data = results.(problemName).(algorithmName).HV;
        validData = data(~isnan(data));
        if ~isempty(validData)
            boxplotData{end+1} = validData;
            boxplotLabels{end+1} = sprintf('%s\n%s', problemName, algorithmName);
        end
    end
end
boxplot(boxplotData, 'Labels', boxplotLabels);
title('Hypervolume Distribution');
xlabel('Problem-Algorithm');
ylabel('HV');
xtickangle(45);
grid on;

% Runtime comparison
subplot(2, 1, 2);
barData = zeros(length(problems), length(algorithms));
for probIdx = 1:length(problems)
    problemName = problems{probIdx};
    for algIdx = 1:length(algorithms)
        algorithmName = algorithms{algIdx};
        meanRuntime = results.(problemName).(algorithmName).Mean_runtime;
        if ~isnan(meanRuntime)
            barData(probIdx, algIdx) = meanRuntime;
        end
    end
end
bar(barData);
legend(algorithms, 'Location', 'best');
title('Average Runtime Comparison');
xlabel('Problem');
ylabel('Runtime (seconds)');
set(gca, 'XTickLabel', problems);
grid on;

%% Statistical significance tests
fprintf('\nPerforming statistical tests...\n');
alpha = 0.05;  % Significance level

for probIdx = 1:length(problems)
    problemName = problems{probIdx};
    fprintf('\n%s - HV Statistical Tests:\n', problemName);
    
    % Compare DiffusionEvolution with other algorithms
    deData = results.(problemName).DiffusionEvolution.HV;
    deData = deData(~isnan(deData));
    
    if ~isempty(deData)
        for algIdx = 2:length(algorithms)
            algorithmName = algorithms{algIdx};
            otherData = results.(problemName).(algorithmName).HV;
            otherData = otherData(~isnan(otherData));
            
            if ~isempty(otherData)
                % Wilcoxon rank-sum test
                [pValue, ~] = ranksum(deData, otherData);
                
                if pValue < alpha
                    fprintf('  DE vs %s: p = %.4f (significant)\n', algorithmName, pValue);
                else
                    fprintf('  DE vs %s: p = %.4f (not significant)\n', algorithmName, pValue);
                end
            end
        end
    end
end

%% Save results
fprintf('\nSaving results...\n');
save('performance_results.mat', 'results');
fprintf('Results saved to performance_results.mat\n');

%% Generate summary report
fprintf('\nGenerating summary report...\n');
fid = fopen('performance_summary.txt', 'w');
fprintf(fid, 'Performance Comparison Summary\n');
fprintf(fid, '=============================\n\n');
fprintf(fid, 'Test Date: %s\n', datestr(now));
fprintf(fid, 'Number of runs: %d\n\n', numRuns);

for probIdx = 1:length(problems)
    problemName = problems{probIdx};
    fprintf(fid, '%s:\n', problemName);
    fprintf(fid, 'Algorithm           HV (mean±std)       Runtime (mean±std)\n');
    fprintf(fid, '--------------------------------------------------------\n');
    
    for algIdx = 1:length(algorithms)
        algorithmName = algorithms{algIdx};
        hvMean = results.(problemName).(algorithmName).Mean_HV;
        hvStd = results.(problemName).(algorithmName).Std_HV;
        runtimeMean = results.(problemName).(algorithmName).Mean_runtime;
        runtimeStd = results.(problemName).(algorithmName).Std_runtime;
        
        fprintf(fid, '%-20s %.4e±%.2e  %.2f±%.2f\n', ...
            algorithmName, hvMean, hvStd, runtimeMean, runtimeStd);
    end
    fprintf(fid, '\n');
end

fclose(fid);
fprintf('Summary report saved to performance_summary.txt\n');

fprintf('\n=== Performance comparison completed ===\n');

end