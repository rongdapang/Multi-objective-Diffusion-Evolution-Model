"""
DM-MOEA主算法模块

该模块实现了基于DDIM扩散模型引导的多目标进化算法（DM-MOEA）。
DM-MOEAEA通过整合DDIM扩散模型的生成能力与进化算法的优化机制，
有效解决多目标优化问题中的探索与利用平衡、算子设计以及避免早熟收敛等挑战。

算法流程：
1. 初始化种群
2. 迭代进化：
   - 评估种群和多样性
   - 根据多样性选择使用DDIM扩散模型或传统算子
   - 环境选择
3. 返回帕帕累托前沿

新特性：
- 多样性驱动的条件生成：将多样性指标作为条件输入，引导生成稀疏区域的解
- 动态调整去噪步数：探索阶段多步，开发阶段少步
- 自适应算子切换：根据多样性水平动态调整算子调用频率
- 智能变异算子：将扩散模型作为智能变异算子
- 帕累托前沿空缺区域生成：根据空缺区域生成解提高分布性
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'operators'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'metrics'))

from individual import Individual
from selection import fast_non_dominated_sort, compute_crowding_distance, environmental_selection
from genetic_operators import GeneticOperators
from diffusion_model import DiffusionModel
from diversity import DiversityMetrics, ConditionEncoder, AdaptiveDiversityThreshold, set_fast_non_dominated_sort_for_encoder


class DM_MOEA:
    """
    基于DDIM扩散模型引导的多目标进化算法
    
    DM-MOEA通过自适应地选择DDIM扩散模型或传统遗传算子来生成新解，
    从而在探索和利用之间取得平衡。
    
    属性:
        problem (Problem): 优化问题对象
        population_size (int): 种群大小
        max_generations (int): 最大迭代次数
        diversity_threshold (float): 多样性阈值
        use_adaptive_threshold (bool): 是否使用自适应阈值
        diffusion_model (DiffusionModel): DDIM扩散模型实例
        genetic_operators (GeneticOperators): 遗传算子实例
        condition_encoder (ConditionEncoder): 条件编码器
        population (list): 当前种群
        best_pareto_front (list): 最优帕累托前沿
        history (dict): 优化历史记录
    """
    
    def __init__(self, problem, config=None):
        """
        初始化DM-MOEA算法
        
        参数:
            problem (Problem): 优化问题对象
            config (dict): 算法配置参数，包括：
                - population_size: 种群大小，默认100
                - max_generations: 最大迭代次数，默认100
                - diversity_threshold: 多样性阈值，默认默认0.5
                - use_adaptive_threshold: 是否使用自适应阈值，默认True
                - diffusion_n_timesteps: 扩散时间步数，默认100
                - diffusion_sample_size: 扩散模型生成样本数，默认20
                - diffusion_noise_scale: 扩散散模型噪声尺度，默认1.0
                - ddim_eta: DDIM的eta参数，控制随机性，默认0.0
                - ddim_perturbation_strength: DDIM扰动强度，默认0.1
                - use_ddim_for_local_search: 是否在多样性良好时使用DDIM进行局部搜索，默认False
                - use_intelligent_mutation: 是否使用智能变异算子，默认True
                - use_pareto_gap_filling: 是否使用帕累托前沿空缺区域生成，默认True
                - crossover_type: 交叉算子类型，默认'sbx'
                - mutation_type: 变异算子类型，默认'polynomial'
                - eta_c: 交叉分布指数，默认20
                - eta_m: 变异分布指数，默认20
                - prob_crossover: 交叉概率，默认0.9
                - prob_mutation: 变异概率，默认为1.0
                - offspring_size: 后代数量，默认等于种群大小
                - train_diffusion_every: 训练扩散模型的频率，默认10
        """
        self.problem = problem
        
        if config is None:
            config = {}
        
        self.population_size = config.get('population_size', 100)
        self.max_generations = config.get('max_generations', 100)
        self.diversity_threshold = config.get('diversity_threshold', 0.5)
        self.use_adaptive_threshold = config.get('use_adaptive_threshold', True)
        
        self.diffusion_n_timesteps = config.get('diffusion_n_timesteps', 100)
        self.diffusion_sample_size = config.get('diffusion_sample_size', 20)
        self.diffusion_noise_scale = config.get('diffusion_noise_scale', 1.0)
        self.ddim_eta = config.get('ddim_eta', 0.0)
        self.ddim_perturbation_strength = config.get('ddim_perturbation_strength', 0.1)
        self.use_ddim_for_local_search = config.get('use_ddim_for_local_search', False)
        self.use_intelligent_mutation = config.get('use_intelligent_mutation', True)
        self.use_pareto_gap_filling = config.get('use_pareto_gap_filling', True)
        
        self.crossover_type = config.get('crossover_type', 'sbx')
        self.mutation_type = config.get('mutation_type', 'polynomial')
        self.eta_c = config.get('eta_c', 20)
        self.eta_m = config.get('eta_m', 20)
        self.prob_crossover = config.get('prob_crossover', 0.9)
        self.prob_mutation = config.get('prob_mutation', 1.0)
        
        self.offspring_size = config.get('offspring_size', self.population_size)
        self.train_diffusion_every = config.get('train_diffusion_every', 10)
        
        self.population = []
        self.best_pareto_front = []
        self.history = {
            'diversity': [],
            'hypervolume': [],
            'igd': [],
            'pareto_size': [],
            'operator_usage': []
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """
        初始化算法组件
        
        包括遗传算子、DDIM扩散模型、条件编码器和自适应阈值。
        """
        self.genetic_operators = GeneticOperators(
            bounds=self.problem.bounds,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            eta_c=self.eta_c,
            eta_m=self.eta_m,
            prob_crossover=self.prob_crossover,
            prob_mutation=self.prob_mutation
        )
        
        self.diffusion_model = DiffusionModel(
            n_vars=self.problem.n_vars,
            bounds=self.problem.bounds,
            n_timesteps=self.diffusion_n_timesteps,
            eta=self.ddim_eta
        )
        
        self.condition_encoder = ConditionEncoder(self.problem)
        set_fast_non_dominated_sort_for_encoder(self.condition_encoder, fast_non_dominated_sort)
        
        if self.use_adaptive_threshold:
            self.adaptive_threshold = AdaptiveDiversityThreshold(
                initial_threshold=self.diversity_threshold
            )
        else:
            self.adaptive_threshold = None
    
    def initialize_population(self):
        """
        初始化种群
        
        随机生成初始种群，评估目标函数，并进行非支配排序和拥挤距离计算。
        """
        self.population = []
        
        for _ in range(self.population_size):
            decision_vars = np.array([
                np.random.uniform(lower, upper)
                for lower, upper in self.problem.bounds
            ])
            
            individual = Individual(decision_vars=decision_vars)
            individual.evaluate(self.problem)
            self.population.append(individual)
        
        fronts = fast_non_dominated_sort(self.population)
        
        for front in fronts:
            compute_crowding_distance(front)
        
        self.best_pareto_front = fronts[0] if len(fronts) > 0 else []
    
    def evaluate_population(self, population):
        """
        评估种群中所有个体的目标函数
        
        参数:
            population (list): 个体列表
        """
        for individual in population:
            if individual.objectives is None:
                individual.evaluate(self.problem)
    
    def compute_diversity(self, population):
        """
        计算种群多样性
        
        参数:
            population (list): 个体列表
            
        返回:
            tuple: (diversity, metrics) 综合多样性和各指标度量的字典
        """
        diversity, metrics = DiversityMetrics.compute_combined_diversity(population)
        return diversity, metrics
    
    def determine_exploration_phase(self, generation, diversity):
        """
        确定当前是否为探索阶段
        
        根据代数和多样性水平判断当前阶段。
        早期或多样性低时为探索阶段，后期或多样性高时为开发阶段。
        
        参数:
            generation (int): 当前代数
            diversity (float): 当前多样性值
            
        返回:
            bool: 如果为探索阶段返回True
        """
        early_phase = generation < self.max_generations * 0.3
        low_diversity = diversity < 0.3
        
        return early_phase or low_diversity
    
    def select_operator(self, diversity, generation):
        """
        自适应算子选择
        
        根据种群多样性水平动态选择使用的算子类型。
        当多样性低于某一阈值时，强制使用具有更强探索能力的遗传算子
        或多样性增强型扩散模型生成，以注入新的遗传物质。
        
        参数:
            diversity (float): 当前多样性值
            generation (int): 当前代数
            
        返回:
            str: 算子类型 ('diffusion', 'genetic', 'hybrid', 'intelligent_mutation', 'pareto_gap')
        """
        if self.adaptive_threshold is not None:
            threshold = self.adaptive_threshold.get_threshold()
        else:
            threshold = self.diversity_threshold
        
        early_phase = generation < self.max_generations * 0.3
        
        if early_phase:
            return 'genetic'
        elif diversity < 0.1:
            return 'diffusion'
        elif diversity < 0.3:
            if generation % 3 == 0:
                return 'diffusion'
            else:
                return 'genetic'
        elif diversity < 0.5:
            if self.use_pareto_gap_filling and generation % 5 == 0:
                return 'pareto_gap'
            else:
                return 'genetic'
        elif diversity < 0.7:
            if self.use_intelligent_mutation and generation % 4 == 0:
                return 'intelligent_mutation'
            else:
                return 'genetic'
        else:
            if self.use_ddim_for_local_search and generation % 3 == 0:
                return 'hybrid'
            else:
                return 'genetic'
    
    def generate_diffusion_offspring(self, population, generation, diversity_metrics, exploration_phase):
        """
        使用DDIM扩散模型生成后代
        
        当多样性不足时，使用DDIM从高斯噪声中生成多样性增强解。
        DDIM的确定性使得对生成过程的控制更为精确。
        使用混合策略结合遗传算子和扩散模型。
        
        参数:
            population (list): 当前种群
            generation (int): 当前代数
            diversity_metrics (dict): 多样性指标
            exploration_phase (bool): 是否为探索阶段
            
        返回:
            list: 生成的后代个体列表
        """
        condition = self.condition_encoder.encode(
            population, generation, self.max_generations
        )
        
        fronts = fast_non_dominated_sort(population)
        pareto_front = fronts[0] if len(fronts) > 0 else []
        
        combined_diversity = diversity_metrics.get('combined_diversity', 0.5)
        
        if combined_diversity < 0.2:
            genetic_ratio = 0.3
        elif combined_diversity < 0.4:
            genetic_ratio = 0.5
        elif combined_diversity < 0.6:
            genetic_ratio = 0.7
        else:
            genetic_ratio = 0.8
        
        n_genetic = int(self.diffusion_sample_size * genetic_ratio)
        n_diffusion = self.diffusion_sample_size - n_genetic
        
        genetic_offspring = self.genetic_operators.generate_offspring(
            population, n_genetic
        )
        
        samples = self.diffusion_model.sample(
            condition=condition,
            n_samples=n_diffusion,
            noise_scale=self.diffusion_noise_scale,
            starting_points=None,
            perturbation_strength=self.ddim_perturbation_strength,
            diversity_metrics=diversity_metrics,
            exploration_phase=exploration_phase,
            elite_solutions=pareto_front
        )
        
        offspring = []
        for individual in genetic_offspring:
            offspring.append(individual)
        
        for sample in samples:
            individual = Individual(decision_vars=sample)
            offspring.append(individual)
        
        return offspring
    
    def generate_intelligent_mutation_offspring(self, population, diversity_metrics):
        """
        使用智能变异算子生成后代
        
        将扩散模型作为智能变异算子，对种群中的个体进行有偏的扰动，
        引导其向多样性更优的方向发展，而不是完全从噪声中生成新解。
        
        参数:
            population (list): 当前种群
            diversity_metrics (dict): 多样性指标
            
        返回:
            list: 生成的后代个体列表
        """
        offspring = []
        
        selected_parents = np.random.choice(population, min(self.offspring_size, len(population)), replace=True)
        
        for parent in selected_parents:
            mutated_vars = self.diffusion_model.intelligent_mutation(
                parent, population, diversity_metrics
            )
            
            individual = Individual(decision_vars=mutated_vars)
            offspring.append(individual)
        
        return offspring
    
    def generate_pareto_gap_filling_offspring(self, population, pareto_front):
        """
        使用帕累托前沿空缺区域生成后代
        
        根据当前帕累托前沿的空缺区域，生成填补这些空缺的解，
        从而提高解的分布性。
        
        参数:
            population (list): 当前种群
            pareto_front (list): 当前帕累托前沿
            
        返回:
            list: 生成的后代个体列表
        """
        if len(pareto_front) < 2:
            return []
        
        n_samples = min(self.diffusion_sample_size // 2, 10)
        
        samples = self.diffusion_model.generate_for_pareto_gaps(
            pareto_front, n_samples=n_samples
        )
        
        offspring = []
        for sample in samples:
            individual = Individual(decision_vars=sample)
            offspring.append(individual)
        
        return offspring
    
    def generate_genetic_offspring(self, population, diversity_metrics=None):
        """
        使用遗传算子生成后代
        
        当多样性良好时，使用传统遗传算子生成后代。
        如果启用了智能变异算子，则混合使用传统变异和智能变异。
        
        参数:
            population (list): 当前种群
            diversity_metrics (dict): 多样性指标（可选）
            
        返回:
            list: 生成的后代个体列表
        """
        offspring = self.genetic_operators.generate_offspring(
            population, self.offspring_size
        )
        
        return offspring
    
    def generate_hybrid_offspring(self, population, pareto_front, diversity_metrics, exploration_phase):
        """
        使用混合策略生成后代
        
        结合传统遗传算子和DDIM局部搜索。
        
        参数:
            population (list): 当前种群
            pareto_front (list): 当前帕累托前沿
            diversity_metrics (dict): 多样性指标
            exploration_phase (bool): 是否为探索阶段
            
        返回:
            list: 生成的后代个体列表
        """
        offspring = self.genetic_operators.generate_offspring(
            population, self.offspring_size // 2
        )
        
        if len(pareto_front) > 0:
            elite_solutions = [ind.decision_vars for ind in pareto_front]
            
            ddim_samples = self.diffusion_model.sample(
                condition=None,
                n_samples=min(10, self.offspring_size // 2),
                noise_scale=0.5,
                starting_points=elite_solutions,
                perturbation_strength=self.ddim_perturbation_strength,
                exploration_phase=exploration_phase
            )
            
            for sample in ddim_samples:
                individual = Individual(decision_vars=sample)
                offspring.append(individual)
        
        return offspring
    
    def train_diffusion_model(self, population):
        """
        训练DDIM扩散模型
        
        使用当前种群的数据训练DDIM扩散模型和条件编码器。
        
        参数:
            population (list): 当前种群
        """
        if len(population) < 2:
            return
        
        decision_vars = [ind.decision_vars for ind in population]
        objectives = [ind.objectives for ind in population]
        
        self.diffusion_model.train(decision_vars, objectives)
        
        self.condition_encoder.fit(population)
    
    def run(self, verbose=True):
        """
        运行DM-MOEA算法
        
        参数:
            verbose (bool): 是否打印详细信息，默认True
            
        返回:
            list: 最终的帕累托前沿
        """
        start_time = time.time()
        
        self.initialize_population()
        
        if verbose:
            print(f"Generation 0: Population size = {len(self.population)}, Pareto front size = {len(self.best_pareto_front)}")
        
        prev_diversity = 0.0
        
        for generation in range(1, self.max_generations + 1):
            diversity, metrics = self.compute_diversity(self.population)
            
            self.history['diversity'].append(diversity)
            self.history['pareto_size'].append(len(self.best_pareto_front))
            
            diversity_improved = diversity > prev_diversity
            
            if self.adaptive_threshold is not None:
                self.adaptive_threshold.update(diversity, diversity_improved)
            
            prev_diversity = diversity
            
            exploration_phase = self.determine_exploration_phase(generation, diversity)
            
            operator_type = self.select_operator(diversity, generation)
            
            self.history['operator_usage'].append(operator_type)
            
            offspring = []
            
            if operator_type == 'diffusion':
                diffusion_offspring = self.generate_diffusion_offspring(
                    self.population, generation, metrics, exploration_phase
                )
                offspring.extend(diffusion_offspring)
                
                if verbose:
                    print(f"  Generation {generation}: Using diffusion model (diversity = {diversity:.4f})")
            
            elif operator_type == 'intelligent_mutation':
                intelligent_offspring = self.generate_intelligent_mutation_offspring(
                    self.population, metrics
                )
                offspring.extend(intelligent_offspring)
                
                if verbose:
                    print(f"  Generation {generation}: Using intelligent mutation (diversity = {diversity:.4f})")
            
            elif operator_type == 'pareto_gap':
                pareto_gap_offspring = self.generate_pareto_gap_filling_offspring(
                    self.population, self.best_pareto_front
                )
                offspring.extend(pareto_gap_offspring)
                
                if verbose:
                    print(f"  Generation {generation}: Using Pareto gap filling (diversity = {diversity:.4f})")
            
            elif operator_type == 'hybrid':
                hybrid_offspring = self.generate_hybrid_offspring(
                    self.population, self.best_pareto_front, metrics, exploration_phase
                )
                offspring.extend(hybrid_offspring)
                
                if verbose:
                    print(f"  Generation {generation}: Using hybrid operators (diversity = {diversity:.4f})")
            
            else:
                genetic_offspring = self.generate_genetic_offspring(self.population, metrics)
                offspring.extend(genetic_offspring)
                
                if verbose:
                    print(f"  Generation {generation}: Using genetic operators (diversity = {diversity:.4f})")
            
            if generation % self.train_diffusion_every == 0:
                self.train_diffusion_model(self.population)
            
            combined_population = self.population + offspring
            
            self.evaluate_population(combined_population)
            
            fronts = fast_non_dominated_sort(combined_population)
            
            for front in fronts:
                compute_crowding_distance(front)
            
            self.population = environmental_selection(combined_population, self.population_size)
            
            self.best_pareto_front = fronts[0] if len(fronts) > 0 else []
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Population size = {len(self.population)}, Pareto front size = {len(self.best_pareto_front)}, Diversity = {diversity:.4f}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if verbose:
            print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
            print(f"Final Pareto front size: {len(self.best_pareto_front)}")
            
            operator_counts = {}
            for op in self.history['operator_usage']:
                operator_counts[op] = operator_counts.get(op, 0) + 1
            
            print("\nOperator usage statistics:")
            for op, count in operator_counts.items():
                print(f"  {op}: {count} times ({count / len(self.history['operator_usage']) * 100:.1f}%)")
        
        return self.best_pareto_front
    
    def get_pareto_front(self):
        """
        获取帕累托前沿
        
        返回:
            list: 帕累托前沿个体列表
        """
        return self.best_pareto_front
    
    def get_pareto_front_objectives(self):
        """
        获取帕累托前沿的目标函数值
        
        返回:
            np.ndarray: 帕累托前沿的目标函数值矩阵
        """
        if len(self.best_pareto_front) == 0:
            return np.array([])
        
        objectives = np.array([ind.objectives for ind in self.best_pareto_front])
        
        objectives = objectives[np.argsort(objectives[:, 0])]
        
        return objectives
