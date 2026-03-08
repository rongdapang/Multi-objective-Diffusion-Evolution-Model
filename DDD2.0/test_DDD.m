%% DDD Algorithm Test Script
% This script tests the DDD algorithm to verify bug fixes
%
% Run this in MATLAB with PlatEMO installed

clc;
clear;

%% Add paths
try
    addpath(genpath('PlatEMO'));
    addpath(pwd);  % Add current directory (DDD_Fixed)
catch
    warning('PlatEMO path not found. Please add PlatEMO to path manually.');
end

%% Test 1: Basic functionality without Deep Learning Toolbox
disp('========================================');
disp('Test 1: Basic Functionality (GA-only mode)');
disp('========================================');
try
    % This should work even without Deep Learning Toolbox
    result = platemo('algorithm', @DDD, ...
                     'problem', @ZDT1, ...
                     'N', 50, ...
                     'maxFE', 1000, ...
                     'save', 0);
    
    % Check result
    if ~isempty(result) && iscell(result) && ~isempty(result{end})
        finalPop = result{end};
        fprintf('✓ Test 1 PASSED: Algorithm completed successfully\n');
        fprintf('  Final population size: %d\n', length(finalPop));
        fprintf('  Number of objectives: %d\n', size(finalPop.objs, 2));
    else
        fprintf('✗ Test 1 FAILED: Empty result\n');
    end
catch ME
    fprintf('✗ Test 1 FAILED: %s\n', ME.message);
    fprintf('Error at: %s\n', ME.stack(1).name);
end

%% Test 2: Check offspring generation
disp(' ');
disp('========================================');
disp('Test 2: Offspring Generation');
disp('========================================');
try
    % Create a mock problem
    prob = ZDT1();
    prob.N = 50;
    
    % Create algorithm instance
    alg = DDD();
    alg.Archive = SolutionArchive(100, 2);
    alg.Scheduler = AdaptiveScheduler(0.4, 15, 50);
    alg.Generation = 10;
    alg.DMStats = struct('total', 0, 'survived', 0, 'history', []);
    alg.DMModel = [];  % No DM model (GA-only mode)
    
    % Create a small population
    pop = prob.Initialization();
    
    % Test GenerateOffspring
    [gaOff, dmOff] = alg.GenerateOffspring(prob, pop);
    
    fprintf('✓ Test 2 PASSED: Offspring generation successful\n');
    fprintf('  GA Offspring: %d\n', length(gaOff));
    fprintf('  DM Offspring: %d\n', length(dmOff));
catch ME
    fprintf('✗ Test 2 FAILED: %s\n', ME.message);
    fprintf('Error at: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

%% Test 3: Check EnvironmentalSelection
disp(' ');
disp('========================================');
disp('Test 3: Environmental Selection');
disp('========================================');
try
    prob = ZDT1();
    prob.N = 50;
    
    % Create combined population
    pop1 = prob.Initialization();
    pop2 = prob.Initialization();
    combined = [pop1, pop2];
    
    % Test EnvironmentalSelection
    [newPop, frontNo, crowdDis] = DDD.EnvironmentalSelection(combined, prob.N);
    
    if length(newPop) == prob.N
        fprintf('✓ Test 3 PASSED: Environmental selection successful\n');
        fprintf('  Selected population size: %d\n', length(newPop));
    else
        fprintf('✗ Test 3 FAILED: Wrong population size (%d instead of %d)\n', length(newPop), prob.N);
    end
catch ME
    fprintf('✗ Test 3 FAILED: %s\n', ME.message);
    fprintf('Error at: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

%% Test 4: Check InitialSampling
disp(' ');
disp('========================================');
disp('Test 4: Initial Sampling');
disp('========================================');
try
    prob = ZDT1();
    prob.N = 50;
    
    alg = DDD();
    
    % Test InitialSampling
    samplePop = alg.InitialSampling(prob, 5, 100);
    
    if ~isempty(samplePop)
        fprintf('✓ Test 4 PASSED: Initial sampling successful\n');
        fprintf('  Sample population size: %d\n', length(samplePop));
    else
        fprintf('✗ Test 4 FAILED: Empty sample population\n');
    end
catch ME
    fprintf('✗ Test 4 FAILED: %s\n', ME.message);
    fprintf('Error at: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

%% Test 5: Check SolutionArchive
disp(' ');
disp('========================================');
disp('Test 5: Solution Archive');
disp('========================================');
try
    prob = ZDT1();
    pop = prob.Initialization();
    
    archive = SolutionArchive(100, 2);
    archive.add(pop);
    
    if archive.Size > 0
        fprintf('✓ Test 5 PASSED: Archive operations successful\n');
        fprintf('  Archive size: %d\n', archive.Size);
        
        % Test getReferencePoints
        refPoints = archive.getReferencePoints(10);
        if ~isempty(refPoints)
            fprintf('  Reference points: %d x %d\n', size(refPoints, 1), size(refPoints, 2));
        end
    else
        fprintf('✗ Test 5 FAILED: Archive is empty\n');
    end
catch ME
    fprintf('✗ Test 5 FAILED: %s\n', ME.message);
    fprintf('Error at: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

%% Summary
disp(' ');
disp('========================================');
disp('Test Summary');
disp('========================================');
disp('If all tests passed, the DDD algorithm should work correctly.');
disp('If any test failed, check the error messages above.');
disp(' ');
disp('To run in PlatEMO GUI:');
disp('  1. Copy DDD_Fixed folder to PlatEMO/Algorithms/Multi-objective optimization/');
disp('  2. Run platemo in MATLAB');
disp('  3. Select DDD algorithm and a test problem');
disp('========================================');
