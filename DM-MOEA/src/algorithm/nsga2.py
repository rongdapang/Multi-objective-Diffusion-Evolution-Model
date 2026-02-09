"""
NSGA-II算法模块

该模块实现了经典的非支配排序遗传算法II（NSGA-II）。
NSGA-II是一种高效的多目标进化算法，通过非支配排序和拥挤距离计算
来维护种群多样性。
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'operators'))

from individual import Individual
from selection import fast_non_dominated_sort, compute_crowding_distance, environmental_selection
from genetic_operators import GeneticOperators


class NSGA2:
    """
    非支配排序遗传算法II（NSGA-II）
    
    NSGA-II是一种经典的多目标进化算法，通过非支配排序和拥挤距离计算
    来维护种群多样性。
    
    属性:
        problem (Problem): 优化问题对象
        population_size (int): 种群大小
        max_generations (int): 最大迭代次数
        genetic_operators (GeneticOperators): 遗传算子实例
        population (list): 当前种群
        best_pareto_front (list): 最优帕累托前沿
        history (dict): 优化历史记录
    """
    
    def __init__(self, problem, config=None):
        """
        初始化NSGA-II算法
        
        参数:
            problem (Problem): 优化问题对象
            config (dict): 算法配置参数，包括：
                - population_size: 种群大小，默认100
                - max_generations: 最大迭代次数，默认100
                - crossover_type: 交叉算子类型，默认'sbx'
                - mutation_type: 变异算子类型，默认'polynomial'
                - eta_c: 交叉分布指数，默认20
                - eta_m: 变异分布指数，默认20
                - prob_crossover: 交叉概率，默认0.9
                - prob_mutation: 变异概率，默认1.0
        """
        self.problem = problem
        
        if config is None:
            config = {}
        
        self.population_size = config.get('population_size', 100)
        self.max_generations = config.get('max_generations', 100)
        
        self.crossover_type = config.get('crossover_type', 'sbx')
        self.mutation_type = config.get('mutation_type', 'polynomial')
        self.eta_c = config.get('eta_c', 20)
        self.eta_m = config.get('eta_m', 20)
        self.prob_crossover = config.get('prob_crossover', 0.9)
        self.prob_mutation = config.get('prob_mutation', 1.0)
        
        self.population = []
        self.best_pareto_front = []
        self.history = {
            'pareto_size': []
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """
        初始化算法组件
        
        包括遗传算子。
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
    
    def generate_offspring(self, population):
        """
        使用遗传算子生成后代
        
        参数:
            population (list): 当前种群
            
        返回:
            list: 生成的后代个体列表
        """
        offspring = self.genetic_operators.generate_offspring(
            population, self.population_size
        )
        return offspring
    
    def run(self, verbose=True):
        """
        运行NSGA-II算法
        
        参数:
            verbose (bool): 是否打印详细信息，默认True
            
        返回:
            list: 最终的帕累托前沿
        """
        start_time = time.time()
        
        self.initialize_population()
        
        if verbose:
            print(f"Generation 0: Population size = {len(self.population)}, Pareto front size = {len(self.best_pareto_front)}")
        
        for generation in range(1, self.max_generations + 1):
            offspring = self.generate_offspring(self.population)
            
            combined_population = self.population + offspring
            
            self.evaluate_population(combined_population)
            
            fronts = fast_non_dominated_sort(combined_population)
            
            for front in fronts:
                compute_crowding_distance(front)
            
            self.population = environmental_selection(combined_population, self.population_size)
            
            self.best_pareto_front = fronts[0] if len(fronts) > 0 else []
            
            self.history['pareto_size'].append(len(self.best_pareto_front))
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Population size = {len(self.population)}, Pareto front size = {len(self.best_pareto_front)}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if verbose:
            print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
            print(f"Final Pareto front size: {len(self.best_pareto_front)}")
        
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
