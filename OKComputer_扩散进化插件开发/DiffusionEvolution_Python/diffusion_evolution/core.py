"""
DiffusionEvolution 核心算法实现

这是将扩散模型集成到进化算法框架中的主要实现。
该算法学习当前种群的分布特征，并使用扩散模型生成高质量的 offspring。
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

from .diffusion import DiffusionModel, NoiseScheduler
from .selection import EnvironmentalSelection, TournamentSelection
from .operators import OperatorGA
from .utils import normalize, denormalize, calculate_diversity


@dataclass
class DiffusionEvolutionConfig:
    """扩散进化算法配置参数"""
    
    # 种群参数
    population_size: int = 100              # 种群大小
    
    # 扩散模型参数
    diffusion_steps: int = 1000             # 扩散步数
    sample_size: int = 50                   # 每代采样数量
    hybrid_rate: float = 0.3                # 扩散解比例
    model_type: str = 'DDPM'                # 扩散模型类型 ('DDPM' 或 'DDIM')
    training_epochs: int = 10               # 每代训练轮数
    noise_schedule: str = 'linear'          # 噪声调度 ('linear' 或 'cosine')
    condition_type: str = 'fitness'         # 条件类型 ('none', 'fitness', 'rank')
    
    # 高级参数
    adaptive_diffusion: bool = True         # 自适应扩散强度
    memory_size: int = 1000                 # 训练数据最大容量
    diffusion_strength: float = 0.1         # 基础扩散强度
    min_hybrid_rate: float = 0.1            # 最小混合比例
    max_hybrid_rate: float = 0.5            # 最大混合比例
    
    # 进化参数
    crossover_prob: float = 1.0             # 交叉概率
    mutation_prob: float = None             # 变异概率 (默认 1/D)
    eta_crossover: float = 20.0             # 交叉分布指数
    eta_mutation: float = 20.0              # 变异分布指数
    tournament_size: int = 2                # 锦标赛选择大小


class Individual:
    """个体类 - 表示优化问题的一个解"""
    
    def __init__(self, decision_vars: np.ndarray, objectives: Optional[np.ndarray] = None, 
                 constraints: Optional[np.ndarray] = None):
        """
        初始化个体
        
        参数:
            decision_vars: 决策变量向量
            objectives: 目标函数值
            constraints: 约束违反程度 (负值表示可行)
        """
        self.decision_vars = decision_vars.copy()
        self.objectives = objectives.copy() if objectives is not None else None
        self.constraints = constraints.copy() if constraints is not None else None
        self.rank = None
        self.crowding_distance = None
        self.age = 0
        
    def clone(self) -> 'Individual':
        """克隆个体"""
        new_individual = Individual(self.decision_vars, self.objectives, self.constraints)
        new_individual.rank = self.rank
        new_individual.crowding_distance = self.crowding_distance
        new_individual.age = self.age
        return new_individual


class Population:
    """种群类 - 管理个体集合"""
    
    def __init__(self, individuals: Optional[List[Individual]] = None):
        self.individuals = individuals if individuals is not None else []
        self.history = {
            'best_fitness': [],
            'average_fitness': [],
            'diversity': []
        }
        
    def __len__(self) -> int:
        return len(self.individuals)
        
    def __getitem__(self, index) -> Individual:
        return self.individuals[index]
        
    def append(self, individual: Individual):
        """添加个体到种群"""
        self.individuals.append(individual)
        
    def extend(self, individuals: List[Individual]):
        """扩展种群"""
        self.individuals.extend(individuals)
        
    def get_decision_matrix(self) -> np.ndarray:
        """获取决策变量矩阵"""
        return np.array([ind.decision_vars for ind in self.individuals])
        
    def get_objective_matrix(self) -> np.ndarray:
        """获取目标函数矩阵"""
        return np.array([ind.objectives for ind in self.individuals])
        
    def get_constraint_matrix(self) -> np.ndarray:
        """获取约束矩阵"""
        return np.array([ind.constraints for ind in self.individuals])


class DiffusionEvolution:
    """
    扩散进化算法主类
    
    这是一个将扩散模型与传统进化算法相结合的多目标优化算法。
    算法使用扩散模型学习当前种群的分布特征，并生成高质量的 offspring。
    """
    
    def __init__(self, config: Optional[DiffusionEvolutionConfig] = None):
        """
        初始化扩散进化算法
        
        参数:
            config: 算法配置参数，如果为None则使用默认配置
        """
        self.config = config if config is not None else DiffusionEvolutionConfig()
        
        # 初始化组件
        self.diffusion_model = None
        self.noise_scheduler = None
        self.environmental_selection = EnvironmentalSelection()
        self.tournament_selection = TournamentSelection()
        self.operators = OperatorGA(
            crossover_prob=self.config.crossover_prob,
            mutation_prob=self.config.mutation_prob,
            eta_crossover=self.config.eta_crossover,
            eta_mutation=self.config.eta_mutation
        )
        
        # 状态变量
        self.generation_count = 0
        self.training_data = np.array([])
        self.best_individuals = []
        self.runtime = 0.0
        self.function_evaluations = 0
        
        # 统计信息
        self.statistics = {
            'training_loss': [],
            'population_diversity': [],
            'hybrid_rate_history': []
        }
        
    def solve(self, problem, max_evaluations: int = 10000, max_runtime: float = float('inf'),
              verbose: bool = True) -> Dict[str, Any]:
        """
        运行优化算法
        
        参数:
            problem: 优化问题实例，需要实现以下方法:
                - evaluate(individuals): 评估个体
                - initialize_population(n): 初始化种群
                - lower_bound: 决策变量下界
                - upper_bound: 决策变量上界
                - n_objectives: 目标函数数量
                - n_variables: 决策变量数量
            max_evaluations: 最大评估次数
            max_runtime: 最大运行时间（秒）
            verbose: 是否显示进度信息
            
        返回:
            包含优化结果的字典
        """
        start_time = time.time()
        
        # 初始化
        population = self._initialize_population(problem)
        self._evaluate_population(population, problem)
        
        if verbose:
            print(f"开始优化: 问题维度={problem.n_variables}, 目标数={problem.n_objectives}")
            print(f"种群大小: {len(population)}, 最大评估次数: {max_evaluations}")
        
        # 主优化循环
        while (self.function_evaluations < max_evaluations and 
               (time.time() - start_time) < max_runtime):
            
            self.generation_count += 1
            
            # 训练扩散模型
            if self.generation_count > 1 and len(self.training_data) > 10:
                self._train_diffusion_model(population, problem)
            
            # 生成 offspring
            offspring = self._generate_offspring(population, problem)
            self._evaluate_population(offspring, problem)
            
            # 环境选择
            combined_population = self._combine_populations(population, offspring)
            population = self._environmental_selection(combined_population, 
                                                       self.config.population_size)
            
            # 更新训练数据
            self._update_training_data(population, problem)
            
            # 更新统计信息
            self._update_statistics(population)
            
            # 显示进度
            if verbose and self.generation_count % 10 == 0:
                progress = self.function_evaluations / max_evaluations * 100
                print(f"第{self.generation_count}代: 评估次数={self.function_evaluations} "
                      f"({progress:.1f}%), 时间={time.time()-start_time:.1f}s")
        
        # 准备结果
        self.runtime = time.time() - start_time
        results = self._prepare_results(population, problem)
        
        if verbose:
            print(f"\n优化完成!")
            print(f"总评估次数: {self.function_evaluations}")
            print(f"运行时间: {self.runtime:.2f}秒")
            print(f"最终种群大小: {len(population)}")
        
        return results
        
    def _initialize_population(self, problem) -> Population:
        """初始化种群"""
        individuals = []
        for _ in range(self.config.population_size):
            decision_vars = np.random.uniform(
                problem.lower_bound, 
                problem.upper_bound, 
                problem.n_variables
            )
            individuals.append(Individual(decision_vars))
        return Population(individuals)
        
    def _evaluate_population(self, population: Population, problem):
        """评估种群"""
        if len(population) == 0:
            return
            
        # 提取决策变量
        decision_matrix = population.get_decision_matrix()
        
        # 评估
        results = problem.evaluate(decision_matrix)
        
        # 更新个体信息
        for i, individual in enumerate(population.individuals):
            individual.objectives = results['objectives'][i]
            if 'constraints' in results:
                individual.constraints = results['constraints'][i]
                
        # 更新评估计数
        self.function_evaluations += len(population)
        
    def _train_diffusion_model(self, population: Population, problem):
        """训练扩散模型"""
        if len(population) < 5:
            return
            
        # 准备训练数据
        training_data, conditions = self._prepare_training_data(population, problem)
        
        # 更新训练数据缓冲区
        if self.training_data.size == 0:
            self.training_data = training_data
        else:
            self.training_data = np.vstack([self.training_data, training_data])
            
        # 限制内存使用
        if len(self.training_data) > self.config.memory_size:
            self.training_data = self.training_data[-self.config.memory_size:]
            
        # 初始化扩散模型（首次训练）
        if self.diffusion_model is None:
            self._initialize_diffusion_model(problem)
            
        # 训练模型
        for epoch in range(self.config.training_epochs):
            loss = self._train_epoch(training_data, conditions)
            self.statistics['training_loss'].append(loss)
            
    def _initialize_diffusion_model(self, problem):
        """初始化扩散模型"""
        self.noise_scheduler = NoiseScheduler(
            timesteps=self.config.diffusion_steps,
            schedule_type=self.config.noise_schedule
        )
        
        self.diffusion_model = DiffusionModel(
            input_dim=problem.n_variables,
            noise_scheduler=self.noise_scheduler,
            model_type=self.config.model_type
        )
        
    def _prepare_training_data(self, population: Population, problem) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """准备训练数据"""
        # 归一化决策变量
        decision_matrix = population.get_decision_matrix()
        normalized_data = normalize(decision_matrix, problem.lower_bound, problem.upper_bound)
        
        # 准备条件信息
        conditions = None
        if self.config.condition_type == 'fitness':
            objective_matrix = population.get_objective_matrix()
            # 使用目标函数值作为条件
            conditions = objective_matrix
        elif self.config.condition_type == 'rank':
            # 计算排序
            ranks = self._calculate_ranks(population)
            conditions = ranks.reshape(-1, 1)
            
        return normalized_data, conditions
        
    def _calculate_ranks(self, population: Population) -> np.ndarray:
        """计算个体的排序等级"""
        objective_matrix = population.get_objective_matrix()
        n_individuals = len(population)
        
        # 非支配排序
        ranks = np.zeros(n_individuals, dtype=int)
        dominated_counts = np.zeros(n_individuals, dtype=int)
        
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i != j:
                    # 检查 i 是否被 j 支配
                    if self._dominates(objective_matrix[j], objective_matrix[i]):
                        dominated_counts[i] += 1
                        
        # 分配等级
        current_rank = 1
        remaining = np.arange(n_individuals)
        
        while len(remaining) > 0:
            # 找到当前前沿的个体
            current_front = remaining[dominated_counts[remaining] == 0]
            ranks[current_front] = current_rank
            
            # 更新支配计数
            for i in current_front:
                for j in remaining:
                    if i != j and self._dominates(objective_matrix[i], objective_matrix[j]):
                        dominated_counts[j] -= 1
                        
            # 移除已分配等级的个体
            remaining = remaining[dominated_counts[remaining] > 0]
            current_rank += 1
            
        return ranks
        
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断 obj1 是否支配 obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
        
    def _train_epoch(self, training_data: np.ndarray, conditions: Optional[np.ndarray]) -> float:
        """训练一个epoch"""
        if self.diffusion_model is None:
            return 0.0
            
        # 随机采样批次
        batch_size = min(32, len(training_data))
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        batch_data = training_data[indices]
        
        batch_conditions = None
        if conditions is not None:
            batch_conditions = conditions[indices]
            
        # 训练一步
        loss = self.diffusion_model.train_step(batch_data, batch_conditions)
        return loss
        
    def _generate_offspring(self, population: Population, problem) -> Population:
        """生成 offspring"""
        n_diffusion = int(self.config.hybrid_rate * self.config.population_size)
        n_traditional = self.config.population_size - n_diffusion
        
        offspring_individuals = []
        
        # 扩散生成
        if n_diffusion > 0 and self.generation_count > 1 and self.diffusion_model is not None:
            diffusion_offspring = self._generate_diffusion_offspring(problem, n_diffusion)
            offspring_individuals.extend(diffusion_offspring)
            
        # 传统遗传操作
        if n_traditional > 0:
            traditional_offspring = self._generate_traditional_offspring(population, problem, n_traditional)
            offspring_individuals.extend(traditional_offspring)
            
        return Population(offspring_individuals)
        
    def _generate_diffusion_offspring(self, problem, n_samples: int) -> List[Individual]:
        """使用扩散模型生成 offspring"""
        if self.training_data.size == 0:
            # 后备方案：随机生成
            individuals = []
            for _ in range(n_samples):
                decision_vars = np.random.uniform(
                    problem.lower_bound, problem.upper_bound, problem.n_variables
                )
                individuals.append(Individual(decision_vars))
            return individuals
            
        # 从扩散模型采样
        samples = self._sample_from_diffusion(n_samples, problem.n_variables)
        
        # 反归一化
        samples = denormalize(samples, problem.lower_bound, problem.upper_bound)
        
        # 确保在边界内
        samples = np.clip(samples, problem.lower_bound, problem.upper_bound)
        
        # 创建个体
        individuals = []
        for i in range(n_samples):
            individuals.append(Individual(samples[i]))
            
        return individuals
        
    def _sample_from_diffusion(self, n_samples: int, n_variables: int) -> np.ndarray:
        """从扩散模型采样"""
        if self.diffusion_model is None:
            return np.random.randn(n_samples, n_variables)
            
        return self.diffusion_model.sample(n_samples, n_variables)
        
    def _generate_traditional_offspring(self, population: Population, problem, n_samples: int) -> List[Individual]:
        """使用传统遗传操作生成 offspring"""
        # 锦标赛选择
        mating_pool = self.tournament_selection.select(
            population, n_samples, self.config.tournament_size
        )
        
        # 遗传操作
        offspring_decisions = self.operators.evolve(mating_pool, problem)
        
        # 创建个体
        individuals = []
        for decision_vars in offspring_decisions:
            individuals.append(Individual(decision_vars))
            
        return individuals
        
    def _combine_populations(self, pop1: Population, pop2: Population) -> Population:
        """合并两个种群"""
        combined_individuals = pop1.individuals + pop2.individuals
        return Population(combined_individuals)
        
    def _environmental_selection(self, population: Population, n_select: int) -> Population:
        """环境选择"""
        selected_indices = self.environmental_selection.select(population, n_select)
        selected_individuals = [population.individuals[i] for i in selected_indices]
        return Population(selected_individuals)
        
    def _update_training_data(self, population: Population, problem):
        """更新训练数据"""
        # 选择最优个体用于条件生成
        if len(population) > 0:
            # 选择第一前沿的个体
            selected_population = self._environmental_selection(population, 
                                                               min(10, len(population)))
            self.best_individuals = selected_population.individuals
            
    def _update_statistics(self, population: Population):
        """更新统计信息"""
        # 计算种群多样性
        diversity = calculate_diversity(population)
        self.statistics['population_diversity'].append(diversity)
        
        # 记录混合比例
        self.statistics['hybrid_rate_history'].append(self.config.hybrid_rate)
        
    def _prepare_results(self, population: Population, problem) -> Dict[str, Any]:
        """准备优化结果"""
        return {
            'population': population,
            'decision_vars': population.get_decision_matrix(),
            'objectives': population.get_objective_matrix(),
            'constraints': population.get_constraint_matrix(),
            'n_evaluations': self.function_evaluations,
            'runtime': self.runtime,
            'statistics': self.statistics.copy(),
            'config': self.config
        }