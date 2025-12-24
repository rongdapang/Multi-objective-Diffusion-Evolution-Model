%% Test Script for DiffusionEvolution Algorithm
% This script tests the DiffusionEvolution algorithm with various problems
% and configurations to ensure proper functionality.
%
% Run this script to validate the implementation before using in production.

%% Test Setup
fprintf('=== DiffusionEvolution Test Suite ===\n\n');

% Test configuration
addpath(genpath('..'));  % Add parent directories to path

% Test results storage
testResults = struct();
testResults.passed = 0;
testResults.failed = 0;
testResults.details = {};

%% Test 1: Algorithm Instantiation
fprintf('Test 1: Algorithm Instantiation... ');
try
    Algorithm = DiffusionEvolution();
    fprintf('PASSED\n');
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'Algorithm instantiation: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('Algorithm instantiation: FAILED - %s', ME.message);
end

%% Test 2: Parameter Configuration
fprintf('\nTest 2: Parameter Configuration... ');
try
    Algorithm = DiffusionEvolution('parameter', {50, 100, 20, 0.3, 'DDPM', 5, 'linear', 'fitness'});
    fprintf('PASSED\n');
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'Parameter configuration: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('Parameter configuration: FAILED - %s', ME.message);
end

%% Test 3: Simple Problem (ZDT1)
fprintf('\nTest 3: Simple Problem (ZDT1)... ');
try
    Problem = ZDT1();
    Algorithm = DiffusionEvolution('parameter', {30, 50, 10, 0.3, 'DDPM', 3, 'linear', 'none'});
    Algorithm.Solve(Problem);
    
    % Check if optimization completed
    if isempty(Algorithm.result)
        error('No results generated');
    end
    
    % Check population size
    if length(Algorithm.result{end}{2}) < 10
        error('Population size too small');
    end
    
    fprintf('PASSED\n');
    fprintf('  - Population size: %d\n', length(Algorithm.result{end}{2}));
    fprintf('  - Evaluations: %d\n', Algorithm.pro.FE);
    fprintf('  - Runtime: %.2f seconds\n', Algorithm.metric.runtime);
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'ZDT1 optimization: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('ZDT1 optimization: FAILED - %s', ME.message);
end

%% Test 4: Problem with Constraints
fprintf('\nTest 4: Constrained Problem (CONSTR)... ');
try
    Problem = CONSTR();
    Algorithm = DiffusionEvolution('parameter', {30, 50, 10, 0.2, 'DDPM', 3, 'linear', 'fitness'});
    Algorithm.Solve(Problem);
    
    % Check if optimization completed
    if isempty(Algorithm.result)
        error('No results generated');
    end
    
    fprintf('PASSED\n');
    fprintf('  - Population size: %d\n', length(Algorithm.result{end}{2}));
    fprintf('  - Evaluations: %d\n', Algorithm.pro.FE);
    
    % Check feasible rate
    Population = Algorithm.result{end}{2};
    feasible = [Population.cons] <= 0;
    feasibleRate = sum(all(feasible, 2)) / length(Population);
    fprintf('  - Feasible rate: %.2f%%\n', feasibleRate * 100);
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'CONSTR optimization: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('CONSTR optimization: FAILED - %s', ME.message);
end

%% Test 5: Many-objective Problem
fprintf('\nTest 5: Many-objective Problem (DTLZ2)... ');
try
    Problem = DTLZ2(3, 5);  % 3 objectives, 5 decision variables
    Algorithm = DiffusionEvolution('parameter', {40, 80, 15, 0.3, 'DDPM', 3, 'cosine', 'rank'});
    Algorithm.Solve(Problem);
    
    % Check if optimization completed
    if isempty(Algorithm.result)
        error('No results generated');
    end
    
    fprintf('PASSED\n');
    fprintf('  - Population size: %d\n', length(Algorithm.result{end}{2}));
    fprintf('  - Evaluations: %d\n', Algorithm.pro.FE);
    fprintf('  - Runtime: %.2f seconds\n', Algorithm.metric.runtime);
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'DTLZ2 optimization: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('DTLZ2 optimization: FAILED - %s', ME.message);
end

%% Test 6: Diffusion Model Training
fprintf('\nTest 6: Diffusion Model Training... ');
try
    % Create a small population for testing
    Problem = ZDT1();
    Population = Problem.Initialization(20);
    
    % Create algorithm instance
    Algorithm = DiffusionEvolution();
    Algorithm.generationCount = 2;  % To trigger training
    Algorithm.initializeDiffusionModel(Problem, 50, 'linear');
    
    % Test training
    Algorithm.trainDiffusionModel(Population, Problem, 2, 'fitness');
    
    % Check if training data was updated
    if isempty(Algorithm.trainingData)
        error('Training data not updated');
    end
    
    fprintf('PASSED\n');
    fprintf('  - Training data size: %d\n', size(Algorithm.trainingData, 1));
    fprintf('  - Last training loss: %.4e\n', Algorithm.diffusionModel.trainingLoss(end));
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'Diffusion model training: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('Diffusion model training: FAILED - %s', ME.message);
end

%% Test 7: Diffusion Sampling
fprintf('\nTest 7: Diffusion Sampling... ');
try
    % Create algorithm instance with trained model
    Problem = ZDT1();
    Algorithm = DiffusionEvolution();
    Algorithm.initializeDiffusionModel(Problem, 20, 'linear');
    
    % Add some training data
    Algorithm.trainingData = rand(10, Problem.D);
    
    % Test sampling
    samples = Algorithm.sampleFromDiffusion(5, 10, 'none');
    
    % Check sample dimensions
    if size(samples, 1) ~= 5 || size(samples, 2) ~= Problem.D
        error('Sample dimensions incorrect');
    end
    
    % Check sample range (should be roughly in [-3, 3] due to normalization)
    if any(samples(:) < -5) || any(samples(:) > 5)
        warning('Sample values outside expected range');
    end
    
    fprintf('PASSED\n');
    fprintf('  - Sample shape: %d x %d\n', size(samples, 1), size(samples, 2));
    fprintf('  - Sample range: [%.2f, %.2f]\n', min(samples(:)), max(samples(:)));
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'Diffusion sampling: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('Diffusion sampling: FAILED - %s', ME.message);
end

%% Test 8: Hybrid Offspring Generation
fprintf('\nTest 8: Hybrid Offspring Generation... ');
try
    Problem = ZDT1();
    Population = Problem.Initialization(30);
    
    Algorithm = DiffusionEvolution();
    Algorithm.generationCount = 2;  % To enable diffusion
    Algorithm.initializeDiffusionModel(Problem, 50, 'linear');
    Algorithm.trainingData = rand(20, Problem.D);  % Mock training data
    
    % Generate hybrid offspring
    Offspring = Algorithm.generateOffspring(Population, Problem, 20, 10, 0.3, 'none');
    
    % Check offspring generation
    if isempty(Offspring)
        error('No offspring generated');
    end
    
    fprintf('PASSED\n');
    fprintf('  - Offspring size: %d\n', length(Offspring));
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'Hybrid offspring generation: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('Hybrid offspring generation: FAILED - %s', ME.message);
end

%% Test 9: Performance Comparison
fprintf('\nTest 9: Performance Comparison with NSGA-II... ');
try
    Problem = ZDT1();
    
    % Run DiffusionEvolution
    DE_Algorithm = DiffusionEvolution('parameter', {50, 100, 20, 0.3, 'DDPM', 5, 'linear', 'fitness'});
    DE_Algorithm.Solve(Problem);
    
    % Run NSGA-II
    NSGA_Algorithm = NSGAII();
    NSGA_Algorithm.Solve(Problem);
    
    % Calculate HV for both
    DE_HV = HV(DE_Algorithm.result{end}{2});
    NSGA_HV = HV(NSGA_Algorithm.result{end}{2});
    
    fprintf('PASSED\n');
    fprintf('  - DiffusionEvolution HV: %.4e\n', DE_HV);
    fprintf('  - NSGA-II HV: %.4e\n', NSGA_HV);
    fprintf('  - DE Runtime: %.2f seconds\n', DE_Algorithm.metric.runtime);
    fprintf('  - NSGA Runtime: %.2f seconds\n', NSGA_Algorithm.metric.runtime);
    testResults.passed = testResults.passed + 1;
    testResults.details{end+1} = 'Performance comparison: PASSED';
catch ME
    fprintf('FAILED\n');
    fprintf('Error: %s\n', ME.message);
    testResults.failed = testResults.failed + 1;
    testResults.details{end+1} = sprintf('Performance comparison: FAILED - %s', ME.message);
end

%% Test Summary
fprintf('\n=== Test Summary ===\n');
fprintf('Total tests: %d\n', testResults.passed + testResults.failed);
fprintf('Passed: %d\n', testResults.passed);
fprintf('Failed: %d\n', testResults.failed);
fprintf('Success rate: %.1f%%\n', testResults.passed / (testResults.passed + testResults.failed) * 100);

fprintf('\n=== Detailed Results ===\n');
for i = 1:length(testResults.details)
    fprintf('%d. %s\n', i, testResults.details{i});
end

%% Save test results
save('test_results.mat', 'testResults');
fprintf('\nTest results saved to test_results.mat\n');
fprintf('\n=== Test suite completed ===\n');

end