# Quick Start Guide

## 5-Minute Quick Start

Get up and running with DiffusionEvolution in just 5 minutes!

### Step 1: Installation (2 minutes)

1. **Download** the DiffusionEvolution plugin
2. **Copy** to your PlatEMO directory:
   ```bash
   cp -r DiffusionEvolution /path/to/PlatEMO/PlatEMO/Algorithms/Multi-objective\ optimization/
   ```
3. **Add to MATLAB path**:
   ```matlab
   addpath('/path/to/PlatEMO');
   addpath('/path/to/PlatEMO/PlatEMO/Algorithms/Multi-objective optimization/DiffusionEvolution');
   ```

### Step 2: First Optimization (2 minutes)

Run your first optimization with just 3 lines of code:

```matlab
% Create problem
Problem = ZDT1();

% Create algorithm
Algorithm = DiffusionEvolution();

% Run optimization
Algorithm.Solve(Problem);

% Show results
fprintf('Optimization complete!\n');
fprintf('Population size: %d\n', length(Algorithm.result{end}{2}));
fprintf('Runtime: %.2f seconds\n', Algorithm.metric.runtime);
```

### Step 3: Visualize Results (1 minute)

```matlab
% Plot the Pareto front
figure;
Draw(Algorithm.result{end}{2}.objs, 'ro');
title('DiffusionEvolution on ZDT1');
xlabel('f_1'); ylabel('f_2');
grid on;
```

That's it! You've successfully run your first DiffusionEvolution optimization.

## Common Usage Patterns

### Pattern 1: Quick Setup

```matlab
% Use default settings (works well for most problems)
Algorithm = DiffusionEvolution();
Algorithm.Solve(Problem);
```

### Pattern 2: Custom Population Size

```matlab
% Larger population for better diversity
Algorithm = DiffusionEvolution('parameter', {200, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness'});
Algorithm.Solve(Problem);
```

### Pattern 3: Many-Objective Problems

```matlab
% For 3+ objectives, use rank conditioning
Algorithm = DiffusionEvolution('parameter', {200, 1000, 100, 0.4, 'DDPM', 15, 'cosine', 'rank'});
Algorithm.Solve(Problem);
```

### Pattern 4: Constrained Problems

```matlab
% Use fitness conditioning for constrained problems
Algorithm = DiffusionEvolution('parameter', {100, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness'});
Algorithm.Solve(Problem);
```

## Parameter Cheat Sheet

### Essential Parameters

| Parameter | When to Change | Recommended Values |
|-----------|----------------|-------------------|
| `N` (Population) | For better diversity | 50-200 |
| `hybrid_rate` | Balance exploration/exploitation | 0.2-0.5 |
| `training_epochs` | Model quality vs speed | 5-20 |
| `condition_type` | Problem type | 'fitness' for constrained |

### Problem-Specific Settings

**Bi-objective problems:**
```matlab
params = {100, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness'};
```

**Many-objective problems (3+ objectives):**
```matlab
params = {200, 1000, 100, 0.4, 'DDPM', 15, 'cosine', 'rank'};
```

**Constrained problems:**
```matlab
params = {100, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness'};
```

**Fast testing:**
```matlab
params = {50, 100, 20, 0.3, 'DDPM', 3, 'linear', 'none'};
```

## Troubleshooting Quick Fixes

### Problem: Slow Performance
**Solution:** Reduce training epochs
```matlab
params = {100, 1000, 50, 0.3, 'DDPM', 3, 'linear', 'fitness'};  % 3 epochs instead of 10
```

### Problem: Poor Convergence
**Solution:** Increase hybrid rate
```matlab
params = {100, 1000, 50, 0.5, 'DDPM', 10, 'linear', 'fitness'};  % 0.5 instead of 0.3
```

### Problem: Out of Memory
**Solution:** Reduce population size
```matlab
params = {50, 1000, 25, 0.3, 'DDPM', 10, 'linear', 'fitness'};  % Smaller population
```

### Problem: Algorithm Crashes
**Solution:** Use simpler configuration
```matlab
params = {50, 100, 20, 0.3, 'DDPM', 3, 'linear', 'none'};  % Minimal configuration
```

## Next Steps

After getting started, explore these resources:

1. **Examples**: See `examples/example_usage.m` for comprehensive examples
2. **Documentation**: Read `README.md` for detailed documentation
3. **Testing**: Run `tests/test_diffusion_evolution.m` to validate installation
4. **Performance**: Run `tests/performance_comparison.m` to compare with other algorithms

## Need Help?

- Check the **FAQ** section in README.md
- Run the test suite to diagnose issues
- Check examples for similar use cases
- Review parameter documentation

## One-Minute Examples

### Example 1: ZDT1 (Bi-objective)
```matlab
Problem = ZDT1();
Algorithm = DiffusionEvolution();
Algorithm.Solve(Problem);
Draw(Algorithm.result{end}{2}.objs, 'ro');
```

### Example 2: DTLZ2 (3 objectives)
```matlab
Problem = DTLZ2(3, 10);
Algorithm = DiffusionEvolution('parameter', {200, 1000, 100, 0.4, 'DDPM', 15, 'cosine', 'rank'});
Algorithm.Solve(Problem);
scatter3(Algorithm.result{end}{2}.objs(:,1), ...
         Algorithm.result{end}{2}.objs(:,2), ...
         Algorithm.result{end}{2}.objs(:,3), 'ro');
```

### Example 3: Compare with NSGA-II
```matlab
Problem = ZDT1();

% DiffusionEvolution
tic; DE = DiffusionEvolution(); DE.Solve(Problem); deTime = toc;

% NSGA-II
tic; NSGA = NSGAII(); NSGA.Solve(Problem); nsgaTime = toc;

% Compare
fprintf('DE Runtime: %.2f, NSGA Runtime: %.2f\n', deTime, nsgaTime);
fprintf('DE HV: %.4e, NSGA HV: %.4e\n', ...
        HV(DE.result{end}{2}), HV(NSGA.result{end}{2}));
```

---

**Ready to optimize? Start with the 5-minute guide above!**