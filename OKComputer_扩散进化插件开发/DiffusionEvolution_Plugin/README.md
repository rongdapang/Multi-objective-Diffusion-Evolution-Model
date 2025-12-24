# DiffusionEvolution: A Hybrid Multi-Objective Optimization Algorithm

## Overview

**DiffusionEvolution** is a novel multi-objective optimization algorithm that integrates diffusion models with traditional evolutionary computation. This algorithm combines the exploratory power of evolutionary algorithms with the generative capabilities of diffusion models to achieve superior performance in complex optimization landscapes.

The algorithm is designed as a plugin for the [PlatEMO](https://github.com/BIMK/PlatEMO) (Platform for Evolutionary Multi-objective Optimization) framework, providing seamless integration with existing optimization tools and benchmarks.

## Key Features

- **Hybrid Architecture**: Combines diffusion models as intelligent mutation operators within evolutionary frameworks
- **Adaptive Learning**: Diffusion model learns population distribution to generate high-quality offspring
- **Flexible Conditioning**: Supports multiple conditioning strategies (fitness-based, rank-based, unconstrained)
- **Multi-Objective Support**: Compatible with bi-objective and many-objective optimization problems
- **Constrained Optimization**: Handles both constrained and unconstrained problems
- **PlatEMO Integration**: Full compatibility with PlatEMO's algorithm interface and problem library

## Algorithm Architecture

The DiffusionEvolution algorithm consists of the following key components:

### 1. **Initialization Module**
- Population initialization using problem-specific methods
- Diffusion model setup with configurable parameters
- Noise schedule initialization (linear or cosine)
- Data normalization for proper diffusion training

### 2. **Diffusion Training Module**
- Trains diffusion model on current population distribution
- Supports conditional generation based on fitness or rank information
- Incremental learning with adaptive memory management
- Configurable training epochs and batch processing

### 3. **Sampling Generation Module**
- Reverse diffusion process for high-quality sample generation
- Supports both DDPM (Denoising Diffusion Probabilistic Models) and DDIM sampling
- Conditional sampling based on problem-specific information
- Batch sampling for efficient offspring generation

### 4. **Hybrid Evolution Module**
- Combines traditional genetic operators with diffusion-based generation
- Configurable hybrid ratio for balancing exploration and exploitation
- Adaptive hybrid rate based on optimization progress
- Tournament selection and environmental selection mechanisms

### 5. **Selection Update Module**
- NSGA-II style environmental selection
- Non-dominated sorting and crowding distance calculation
- Elite preservation for maintaining solution quality
- Dynamic training data management

## Installation

### Prerequisites

- MATLAB R2016b or later
- PlatEMO framework (version 4.0 or later)
- Sufficient memory for population storage and model training

### Installation Steps

1. **Clone or download** the DiffusionEvolution plugin files

2. **Copy the plugin** to your PlatEMO installation directory:
   ```bash
   cp -r DiffusionEvolution /path/to/PlatEMO/Algorithms/Multi-objective\ optimization/
   ```

3. **Add PlatEMO to MATLAB path** (if not already done):
   ```matlab
   addpath('/path/to/PlatEMO');
   addpath('/path/to/PlatEMO/Problems');
   addpath('/path/to/PlatEMO/Algorithms');
   ```

4. **Verify installation** by running a simple test:
   ```matlab
   Algorithm = DiffusionEvolution();
   disp('Installation successful!');
   ```

## Usage

### Basic Usage

```matlab
% Create a problem instance
Problem = ZDT1();

% Create algorithm with default parameters
Algorithm = DiffusionEvolution();

% Run optimization
Algorithm.Solve(Problem);

% Access results
finalPopulation = Algorithm.result{end}{2};
fprintf('Population size: %d\n', length(finalPopulation));
fprintf('Runtime: %.2f seconds\n', Algorithm.metric.runtime);
```

### Advanced Configuration

```matlab
% Configure algorithm parameters
params = {
    100,    % Population size (N)
    1000,   % Diffusion steps
    50,     % Sample size per generation
    0.3,    % Hybrid rate (proportion of diffusion offspring)
    'DDPM', % Model type ('DDPM' or 'DDIM')
    10,     % Training epochs per generation
    'linear', % Noise schedule ('linear' or 'cosine')
    'fitness' % Condition type ('none', 'fitness', 'rank')
};

Algorithm = DiffusionEvolution('parameter', params);
```

### Parameter Descriptions

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `N` | Population size | 100 | [50, 500] |
| `diffusion_steps` | Number of diffusion steps | 1000 | [100, 2000] |
| `sample_size` | Samples generated per generation | 50 | [20, 200] |
| `hybrid_rate` | Proportion of diffusion offspring | 0.3 | [0.0, 1.0] |
| `model_type` | Diffusion model type | 'DDPM' | {'DDPM', 'DDIM'} |
| `training_epochs` | Training epochs per generation | 10 | [1, 50] |
| `noise_schedule` | Noise schedule type | 'linear' | {'linear', 'cosine'} |
| `condition_type` | Condition type | 'fitness' | {'none', 'fitness', 'rank'} |

## Examples

### Example 1: Bi-objective Optimization (ZDT1)

```matlab
% ZDT1 problem (2 objectives, 30 variables)
Problem = ZDT1();

% Configure DiffusionEvolution
Algorithm = DiffusionEvolution('parameter', {100, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness'});

% Run optimization
Algorithm.Solve(Problem);

% Plot results
figure;
Draw(Algorithm.result{end}{2}.objs, 'ro');
title('DiffusionEvolution on ZDT1');
xlabel('f_1'); ylabel('f_2');
grid on;
```

### Example 2: Many-objective Optimization (DTLZ2)

```matlab
% DTLZ2 problem (3 objectives, 10 variables)
Problem = DTLZ2(3, 10);

% Configure for many-objective optimization
Algorithm = DiffusionEvolution('parameter', {200, 1000, 100, 0.4, 'DDPM', 15, 'cosine', 'rank'});

% Run optimization
Algorithm.Solve(Problem);

% Plot 3D objective space
figure;
objs = [Algorithm.result{end}{2}.objs];
scatter3(objs(:,1), objs(:,2), objs(:,3), 'ro', 'filled');
title('DiffusionEvolution on DTLZ2 (3 objectives)');
xlabel('f_1'); ylabel('f_2'); zlabel('f_3');
grid on; view(45, 45);
```

### Example 3: Constrained Optimization (CONSTR)

```matlab
% CONSTR constrained problem
Problem = CONSTR();

% Run with default settings
Algorithm = DiffusionEvolution();
Algorithm.Solve(Problem);

% Plot feasible and infeasible solutions
Population = Algorithm.result{end}{2};
feasible = [Population.cons] <= 0;
feasibleIdx = find(all(feasible, 2));
infeasibleIdx = find(~all(feasible, 2));

figure;
hold on;
if ~isempty(feasibleIdx)
    plot(Population(feasibleIdx).objs(:,1), Population(feasibleIdx).objs(:,2), 'go');
end
if ~isempty(infeasibleIdx)
    plot(Population(infeasibleIdx).objs(:,1), Population(infeasibleIdx).objs(:,2), 'rx');
end
legend('Feasible', 'Infeasible');
title('DiffusionEvolution on CONSTR');
xlabel('f_1'); ylabel('f_2');
grid on; hold off;
```

### Example 4: Comparison with Other Algorithms

```matlab
% Compare with NSGA-II
Problem = ZDT1();

% Run DiffusionEvolution
DE_Algorithm = DiffusionEvolution();
DE_Algorithm.Solve(Problem);

% Run NSGA-II
NSGA_Algorithm = NSGAII();
NSGA_Algorithm.Solve(Problem);

% Plot comparison
figure;
subplot(1,2,1);
Draw(DE_Algorithm.result{end}{2}.objs, 'ro');
title('DiffusionEvolution');
subplot(1,2,2);
Draw(NSGA_Algorithm.result{end}{2}.objs, 'bo');
title('NSGA-II');
```

## Algorithm Components

### Core Files

- **`DiffusionEvolution.m`**: Main algorithm class implementing the hybrid optimization approach
- **`EnvironmentalSelection.m`**: NSGA-II style environmental selection with non-dominated sorting
- **`TournamentSelection.m`**: Tournament selection for mating pool generation
- **`OperatorGA.m`**: Genetic operators (SBX crossover and polynomial mutation)

### Configuration Files

- **`configs/default_config.m`**: Default parameter configuration
- **`configs/test_config.m`**: Test configuration for debugging

### Examples and Tests

- **`examples/example_usage.m`**: Comprehensive usage examples
- **`tests/test_diffusion_evolution.m`**: Validation test suite
- **`tests/performance_comparison.m`**: Performance benchmarking script

## Theory and Methodology

### Diffusion Models in Optimization

Diffusion models are generative models that learn to generate data by reversing a diffusion process - gradually adding noise to data until it becomes nearly pure noise, then learning to reverse this process to reconstruct the original data.

In the context of optimization:

1. **Forward Process**: Population solutions are gradually noised over multiple steps
2. **Reverse Process**: The trained model learns to denoise and generate new solutions
3. **Conditioning**: Additional information (fitness, rank) guides generation toward better regions

### Hybrid Strategy

The algorithm balances exploration and exploitation through:

- **Diffusion Offspring**: Generated by the trained diffusion model for directed exploration
- **Traditional Offspring**: Created using genetic operators for global exploration
- **Adaptive Ratio**: The proportion can be adjusted based on optimization progress

### Conditioning Strategies

- **`none`**: Unconditional generation based on population distribution
- **`fitness`**: Uses objective values to guide generation toward better regions
- **`rank`**: Uses dominance ranks to focus on promising areas of the search space

## Performance Evaluation

### Benchmark Problems

The algorithm has been tested on standard multi-objective optimization benchmarks:

- **ZDT Suite**: ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
- **DTLZ Suite**: DTLZ1-DTLZ7 (scalable to many objectives)
- **Constrained Problems**: CONSTR, SRN, TNK
- **Real-world Applications**: Engineering design problems

### Comparison with State-of-the-Art

Performance metrics compared against:
- NSGA-II (Nondominated Sorting Genetic Algorithm II)
- NSGA-III (Nondominated Sorting Genetic Algorithm III)
- MOEA/D (Multi-objective Evolutionary Algorithm based on Decomposition)
- Other PlatEMO algorithms

### Key Performance Indicators

- **HV (Hypervolume)**: Measures convergence and diversity
- **IGD (Inverted Generational Distance)**: Measures proximity to true Pareto front
- **GD (Generational Distance)**: Measures convergence quality
- **Feasible Rate**: For constrained problems
- **Runtime**: Computational efficiency

## Advanced Features

### Adaptive Diffusion Strength

The algorithm can adapt the diffusion strength based on optimization progress:

- High strength early in optimization for exploration
- Lower strength later for fine-tuning solutions
- Automatic adjustment based on population diversity

### Memory Management

- Limited memory for training data (configurable size)
- Sliding window approach to maintain recent population history
- Automatic cleanup of outdated training samples

### Multiple Noise Schedules

- **Linear Schedule**: Uniform noise addition across steps
- **Cosine Schedule**: More sophisticated schedule for better generation quality

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce population size (`N` parameter)
   - Decrease training data memory size
   - Use smaller problems for testing

2. **Poor Convergence**
   - Increase number of diffusion steps
   - Adjust hybrid rate (try higher values)
   - Increase training epochs
   - Try different noise schedules

3. **Slow Performance**
   - Reduce training epochs per generation
   - Decrease sample size
   - Use smaller population size
   - Disable adaptive features for faster execution

### Debugging Tips

1. **Enable Verbose Output**
   ```matlab
   Algorithm.outputFcn = @DefaultOutput;
   ```

2. **Check Training Progress**
   ```matlab
   plot(Algorithm.diffusionModel.trainingLoss);
   title('Training Loss Over Time');
   ```

3. **Monitor Population Diversity**
   ```matlab
   diversity = calculateDiversity(finalPop);
   fprintf('Population diversity: %.4f\n', diversity);
   ```

## Contributing

We welcome contributions to improve the DiffusionEvolution algorithm! Areas for contribution include:

- **Performance Improvements**: Faster training, better sampling strategies
- **New Features**: Additional conditioning methods, adaptive mechanisms
- **Testing**: More comprehensive test suites, additional benchmark problems
- **Documentation**: Tutorials, examples, theoretical explanations
- **Integration**: Better integration with other PlatEMO features

### Development Guidelines

1. Follow PlatEMO coding conventions
2. Add comprehensive comments and documentation
3. Include unit tests for new features
4. Maintain backward compatibility
5. Test with multiple problems and configurations

## Citation

If you use DiffusionEvolution in your research, please cite:

```bibtex
@software{diffusionevolution2025,
  title={DiffusionEvolution: A Hybrid Multi-Objective Optimization Algorithm},
  author={Developer Name},
  year={2025},
  url={https://github.com/your-repo/DiffusionEvolution},
  note={PlatEMO Plugin}
}
```

And the PlatEMO framework:

```bibtex
@article{platemo2017,
  title={PlatEMO: A MATLAB platform for evolutionary multi-objective optimization},
  author={Tian, Ye and Cheng, Ran and Zhang, Xingyi and Jin, Yaochu},
  journal={IEEE Computational Intelligence Magazine},
  volume={12},
  number={4},
  pages={73--87},
  year={2017}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PlatEMO Team**: For providing the excellent optimization platform
- **Diffusion Model Community**: For foundational research on diffusion models
- **Multi-objective Optimization Community**: For continuous algorithm development and benchmarking

## References

1. **Diffusion Models**
   - Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
   - Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. ICLR.

2. **Evolutionary Multi-objective Optimization**
   - Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE TEC.
   - Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE TEC.

3. **Hybrid Optimization Methods**
   - Various works on combining machine learning with evolutionary algorithms

## Contact

For questions, bug reports, or feature requests:

- **Issues**: [GitHub Issues](https://github.com/your-repo/DiffusionEvolution/issues)
- **Email**: your.email@university.edu
- **Documentation**: [Full Documentation](docs/)

---

**Note**: This is a research implementation. For production use, additional testing and validation are recommended.