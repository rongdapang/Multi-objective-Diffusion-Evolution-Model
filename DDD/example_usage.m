%% DDD Algorithm Usage Examples
% This script demonstrates how to use the DDD algorithm with PlatEMO

%% Setup
% Add PlatEMO to path (adjust path as needed)
% addpath('path/to/PlatEMO');

%% Example 1: Basic Usage
% Run DDD on DTLZ1 problem with default parameters
% main('-algorithm', @DDD, '-problem', @DTLZ1, '-N', 100, '-M', 3, '-D', 10, '-evaluation', 10000);

%% Example 2: Custom Parameters
% Run DDD with custom parameters
% main('-algorithm', {@DDD, [0.1, 0.01], [256, 512, 512, 256], 20, 500, 100, 50, 1000, 10, 0.4, true}, ...
%      '-problem', @DTLZ2, '-N', 100, '-M', 3, '-D', 10, '-evaluation', 10000);

%% Example 3: Compare with NSGA-II
% Run comparison between DDD and NSGA-II
% main('-algorithm', {@DDD, @NSGA2}, '-problem', @DTLZ1, '-N', 100, '-M', 3, '-D', 10, '-evaluation', 10000, '-run', 10);

%% Example 4: High-dimensional Problem
% Test DDD on high-dimensional problem
% main('-algorithm', @DDD, '-problem', @DTLZ4, '-N', 200, '-M', 5, '-D', 50, '-evaluation', 50000);

%% Example 5: Many-objective Problem
% Test DDD on many-objective problem
% main('-algorithm', @DDD, '-problem', @DTLZ7, '-N', 300, '-M', 10, '-D', 20, '-evaluation', 100000);

%% Example 6: Using Simple Version (no Deep Learning Toolbox required)
% Run simplified version for broader compatibility
% main('-algorithm', @DDD_Simple, '-problem', @DTLZ1, '-N', 100, '-M', 3, '-D', 10, '-evaluation', 10000);

%% Example 7: Batch Testing
% Test DDD on multiple problems
problems = {@DTLZ1, @DTLZ2, @DTLZ3, @DTLZ4};
for i = 1:length(problems)
    fprintf('Testing on %s...\n', func2str(problems{i}));
    % main('-algorithm', @DDD, '-problem', problems{i}, '-N', 100, '-M', 3, '-D', 10, '-evaluation', 10000, '-run', 5);
end

%% Example 8: Parameter Sensitivity Analysis
% Test different DM ratios
% dm_ratios = [0.2, 0.3, 0.4, 0.5];
% for ratio = dm_ratios
%     fprintf('Testing with DM ratio = %.1f...\n', ratio);
%     main('-algorithm', {@DDD, [0.1, 0.01], [256, 512, 512, 256], 20, 500, 100, 50, 1000, 10, ratio, true}, ...
%          '-problem', @DTLZ1, '-N', 100, '-M', 3, '-D', 10, '-evaluation', 10000, '-run', 5);
% end

%% Example 9: Convergence Analysis
% Run with detailed output for convergence analysis
% main('-algorithm', @DDD, '-problem', @DTLZ1, '-N', 100, '-M', 3, '-D', 10, ...
%      '-evaluation', 10000, '-outputFcn', @convergenceOutput);

%% Example 10: Save Results
% Save results to file
% result = main('-algorithm', @DDD, '-problem', @DTLZ1, '-N', 100, '-M', 3, '-D', 10, '-evaluation', 10000);
% save('ddd_results.mat', 'result');

%% Helper function for convergence output
function state = convergenceOutput(options, state, flag)
    persistent igd_history;
    
    switch flag
        case 'init'
            igd_history = [];
        case 'iter'
            % Calculate IGD if true Pareto front is available
            if isfield(options, 'truePF')
                igd = IGD(state.Population, options.truePF);
                igd_history = [igd_history; state.Generation, igd];
                fprintf('Generation %d: IGD = %.4f\n', state.Generation, igd);
            end
        case 'done'
            if ~isempty(igd_history)
                figure;
                plot(igd_history(:, 1), igd_history(:, 2), 'b-', 'LineWidth', 2);
                xlabel('Generation');
                ylabel('IGD');
                title('DDD Convergence');
                grid on;
            end
    end
end

%% Print algorithm information
fprintf('\n');
fprintf('========================================\n');
fprintf('DDD Algorithm Usage Examples\n');
fprintf('========================================\n');
fprintf('\n');
fprintf('Available files:\n');
fprintf('  - DDD.m: Main algorithm (full version with OOP)\n');
fprintf('  - DDD_Simple.m: Simplified version (single file)\n');
fprintf('  - SolutionArchive.m: Elite solution archive\n');
fprintf('  - ConditionalDiffusionModel.m: Diffusion model\n');
fprintf('  - AdaptiveScheduler.m: Adaptive DM/GA scheduler\n');
fprintf('  - SinusoidalTimeEmbedding.m: Time embedding utility\n');
fprintf('  - FiLMConditioning.m: FiLM conditioning utility\n');
fprintf('  - ResidualBlock.m: Residual block utility\n');
fprintf('\n');
fprintf('To use the algorithm:\n');
fprintf('  1. Copy DDD folder to PlatEMO/Algorithms/\n');
fprintf('  2. Run: main(''-algorithm'', @DDD, ''-problem'', @DTLZ1, ...)\n');
fprintf('\n');
fprintf('For more details, see README.md\n');
fprintf('========================================\n');
