%% DiffusionEvolution Usage Example
% This example demonstrates how to use the DiffusionEvolution algorithm
% with PlatEMO framework for multi-objective optimization.
%
% Before running this example, ensure that:
% 1. PlatEMO is installed and in MATLAB path
% 2. The DiffusionEvolution plugin is in the PlatEMO/Algorithms directory
% 3. All required files are in the correct locations

%% Clear workspace and add paths
clear all;
clc;

% Add PlatEMO paths (adjust according to your installation)
% addpath('path/to/PlatEMO');
% addpath('path/to/PlatEMO/Problems');
% addpath('path/to/PlatEMO/Algorithms');

%% Example 1: Basic Usage with ZDT1 Problem
fprintf('=== Example 1: DiffusionEvolution on ZDT1 ===\n');

% Create problem instance
Problem = ZDT1();

% Create algorithm instance with default parameters
Algorithm = DiffusionEvolution();

% Run the algorithm
Algorithm.Solve(Problem);

% Display results
fprintf('Optimization completed!\n');
fprintf('Population size: %d\n', length(Algorithm.result{end}{2}));
fprintf('Number of evaluations: %d\n', Algorithm.pro.FE);
fprintf('Runtime: %.2f seconds\n', Algorithm.metric.runtime);

% Plot results
figure;
Draw(Algorithm.result{end}{2}.objs, 'ro');
title('DiffusionEvolution on ZDT1 - Objective Space');
xlabel('f_1');
ylabel('f_2');
grid on;

%% Example 2: Custom Parameters
fprintf('\n=== Example 2: DiffusionEvolution with Custom Parameters ===\n');

% Create problem instance
Problem = ZDT2();

% Create algorithm instance with custom parameters
Algorithm = DiffusionEvolution('parameter', {50, 500, 25, 0.4, 'DDPM', 5, 'linear', 'fitness'});

% Run the algorithm
Algorithm.Solve(Problem);

% Display results
fprintf('Optimization completed!\n');
fprintf('Population size: %d\n', length(Algorithm.result{end}{2}));
fprintf('Number of evaluations: %d\n', Algorithm.pro.FE);
fprintf('Runtime: %.2f seconds\n', Algorithm.metric.runtime);

%% Example 3: Compare with NSGA-II
fprintf('\n=== Example 3: Comparison with NSGA-II ===\n');

% Create problem instance
Problem = ZDT3();

% Run DiffusionEvolution
fprintf('Running DiffusionEvolution...\n');
DE_Algorithm = DiffusionEvolution('parameter', {100, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness'});
DE_Algorithm.Solve(Problem);

% Run NSGA-II
fprintf('Running NSGA-II...\n');
NSGA_Algorithm = NSGAII();
NSGA_Algorithm.Solve(Problem);

% Compare results
figure('Name', 'Algorithm Comparison');
subplot(1,2,1);
Draw(DE_Algorithm.result{end}{2}.objs, 'ro');
title('DiffusionEvolution on ZDT3');
xlabel('f_1');
ylabel('f_2');
grid on;

subplot(1,2,2);
Draw(NSGA_Algorithm.result{end}{2}.objs, 'bo');
title('NSGA-II on ZDT3');
xlabel('f_1');
ylabel('f_2');
grid on;

%% Example 4: Many-objective Problem (DTLZ2)
fprintf('\n=== Example 4: Many-objective Problem (DTLZ2) ===\n');

% Create 3-objective problem
Problem = DTLZ2(3, 10);  % 3 objectives, 10 decision variables

% Configure algorithm for many-objective optimization
Algorithm = DiffusionEvolution('parameter', {200, 1000, 100, 0.4, 'DDPM', 15, 'linear', 'rank'});

% Run the algorithm
Algorithm.Solve(Problem);

% Display results
fprintf('Optimization completed!\n');
fprintf('Population size: %d\n', length(Algorithm.result{end}{2}));
fprintf('Number of evaluations: %d\n', Algorithm.pro.FE);
fprintf('Runtime: %.2f seconds\n', Algorithm.metric.runtime);

% Plot 3D objective space
figure;
objs = [Algorithm.result{end}{2}.objs];
scatter3(objs(:,1), objs(:,2), objs(:,3), 'ro', 'filled');
title('DiffusionEvolution on DTLZ2 (3 objectives)');
xlabel('f_1');
ylabel('f_2');
zlabel('f_3');
grid on;
view(45, 45);

%% Example 5: Constrained Problem
fprintf('\n=== Example 5: Constrained Problem (CONSTR) ===\n');

% Create constrained problem
Problem = CONSTR();

% Run algorithm
Algorithm = DiffusionEvolution();
Algorithm.Solve(Problem);

% Display results
fprintf('Optimization completed!\n');
fprintf('Population size: %d\n', length(Algorithm.result{end}{2}));
fprintf('Number of evaluations: %d\n', Algorithm.pro.FE);
fprintf('Runtime: %.2f seconds\n', Algorithm.metric.runtime);

% Plot results with feasible/infeasible distinction
figure;
Population = Algorithm.result{end}{2};
feasible = [Population.cons] <= 0;
feasibleIdx = find(all(feasible, 2));
infeasibleIdx = find(~all(feasible, 2));

hold on;
if ~isempty(feasibleIdx)
    plot(Population(feasibleIdx).objs(:,1), Population(feasibleIdx).objs(:,2), 'go', 'MarkerSize', 6);
end
if ~isempty(infeasibleIdx)
    plot(Population(infeasibleIdx).objs(:,1), Population(infeasibleIdx).objs(:,2), 'rx', 'MarkerSize', 6);
end
legend('Feasible', 'Infeasible');
title('DiffusionEvolution on CONSTR');
xlabel('f_1');
ylabel('f_2');
grid on;
hold off;

%% Performance Metrics
fprintf('\n=== Performance Metrics ===\n');

% Calculate IGD (Inverted Generational Distance)
% Note: This requires the true Pareto front
% IGD_value = IGD(Algorithm.result{end}{2}, Problem.optimum);
% fprintf('IGD: %.4e\n', IGD_value);

% Calculate HV (Hypervolume)
HV_value = HV(Algorithm.result{end}{2});
fprintf('HV: %.4e\n', HV_value);

% Calculate GD (Generational Distance)
% GD_value = GD(Algorithm.result{end}{2}, Problem.optimum);
% fprintf('GD: %.4e\n', GD_value);

fprintf('\n=== Examples completed! ===\n');

end