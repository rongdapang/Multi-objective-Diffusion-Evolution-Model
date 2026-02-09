"""
遗传算子模块

该模块实现了多目标进化算法中常用的遗传算子，包括：
- 交叉算子（Crossover Operators）：模拟二进制交叉（SBX）、均匀交叉、算术交叉
- 变异算子（Mutation Operators）：多项式变异、高斯变异、均匀变异

这些算子用于生成新的候选解，探索解空间。
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from individual import Individual


class CrossoverOperator:
    """
    交叉算子类
    
    提供多种交叉算子，用于从两个父代个体生成两个子代个体。
    """
    
    @staticmethod
    def sbx_crossover(parent1, parent2, bounds, eta=20, prob_crossover=0.9):
        """
        模拟二进制交叉（Simulated Binary Crossover, SBX）
        
        SBX模拟单点二进制交叉在实数编码中的行为，能够产生接近父代的子代。
        参数eta控制交叉的分布指数，值越大产生的子代越接近父代。
        
        参数:
            parent1 (Individual): 第一个父代个体
            parent2 (Individual): 第二个父代个体
            bounds (list): 决策变量的边界
            eta (float): 分布指数，默认为20
            prob_crossover (float): 交叉概率，默认为0.9
            
        返回:
            tuple: (child1, child2) 两个子代个体
        """
        if np.random.random() > prob_crossover:
            return parent1.copy(), parent2.copy()
        
        child1_vars = parent1.decision_vars.copy()
        child2_vars = parent2.decision_vars.copy()
        
        for i in range(len(parent1.decision_vars)):
            if np.random.random() <= 0.5:
                if abs(parent1.decision_vars[i] - parent2.decision_vars[i]) > 1e-14:
                    y1 = min(parent1.decision_vars[i], parent2.decision_vars[i])
                    y2 = max(parent1.decision_vars[i], parent2.decision_vars[i])
                    
                    lower_bound, upper_bound = bounds[i]
                    rand = np.random.random()
                    
                    if rand <= 0.5:
                        beta = 2.0 - (1.0 + 2.0 * (y1 - lower_bound) / (y2 - y1)) ** (-(eta + 1))
                    else:
                        beta = 2.0 - (1.0 + 2.0 * (upper_bound - y2) / (y2 - y1)) ** (-(eta + 1))
                    
                    alpha = 2.0 - beta ** -(eta + 1)
                    
                    if rand <= 0.5:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = ((1.0 - rand) * alpha) ** (1.0 / (eta + 1))
                    
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    c1 = np.clip(c1, lower_bound, upper_bound)
                    c2 = np.clip(c2, lower_bound, upper_bound)
                    
                    if np.random.random() <= 0.5:
                        child1_vars[i] = c1
                        child2_vars[i] = c2
                    else:
                        child1_vars[i] = c2
                        child2_vars[i] = c1
        
        child1 = Individual(decision_vars=child1_vars)
        child2 = Individual(decision_vars=child2_vars)
        
        return child1, child2
    
    @staticmethod
    def uniform_crossover(parent1, parent2, prob_crossover=0.9):
        """
        均匀交叉（Uniform Crossover）
        
        对于每个决策变量，以0.5的概率从父代1或父代2中选择。
        
        参数:
            parent1 (Individual): 第一个父代个体
            parent2 (Individual): 第二个父代个体
            prob_crossover (float): 交叉概率，默认为0.9
            
        返回:
            tuple: (child1, child2) 两个子代个体
        """
        if np.random.random() > prob_crossover:
            return parent1.copy(), parent2.copy()
        
        child1_vars = []
        child2_vars = []
        
        for i in range(len(parent1.decision_vars)):
            if np.random.random() <= 0.5:
                child1_vars.append(parent1.decision_vars[i])
                child2_vars.append(parent2.decision_vars[i])
            else:
                child1_vars.append(parent2.decision_vars[i])
                child2_vars.append(parent1.decision_vars[i])
        
        child1 = Individual(decision_vars=np.array(child1_vars))
        child2 = Individual(decision_vars=np.array(child2_vars))
        
        return child1, child2
    
    @staticmethod
    def arithmetic_crossover(parent1, parent2, prob_crossover=0.9):
        """
        算术交叉（Arithmetic Crossover）
        
        子代是父代的线性组合：child1 = α*parent1 + (1-α)*parent2
        其中α是[0,1]之间的随机数。
        
        参数:
            parent1 (Individual): 第一个父代个体
            parent2 (Individual): 第二个父代个体
            prob_crossover (float): 交叉概率，默认为0.9
            
        返回:
            tuple: (child1, child2) 两个子代个体
        """
        if np.random.random() > prob_crossover:
            return parent1.copy(), parent2.copy()
        
        alpha = np.random.random()
        
        child1_vars = alpha * parent1.decision_vars + (1 - alpha) * parent2.decision_vars
        child2_vars = (1 - alpha) * parent1.decision_vars + alpha * parent2.decision_vars
        
        child1 = Individual(decision_vars=child1_vars)
        child2 = Individual(decision_vars=child2_vars)
        
        return child1, child2


class MutationOperator:
    """
    变异算子类
    
    提供多种变异算子，用于对个体进行扰动以探索解空间。
    """
    
    @staticmethod
    def polynomial_mutation(individual, bounds, eta=20, prob_mutation=1.0):
        """
        多项式变异（Polynomial Mutation）
        
        多项式变异模拟二进制变异在实数编码中的行为，能够产生接近原值的变异。
        参数eta控制变异的分布指数，值越大产生的变异值越小。
        
        参数:
            individual (Individual): 要变异的个体
            bounds (list): 决策变量的边界
            eta (float): 分布指数，默认为20
            prob_mutation (float): 变异概率，默认为1.0
            
        返回:
            Individual: 变异后的新个体
        """
        child_vars = individual.decision_vars.copy()
        
        for i in range(len(child_vars)):
            if np.random.random() <= prob_mutation / len(child_vars):
                y = child_vars[i]
                lower_bound, upper_bound = bounds[i]
                delta1 = (y - lower_bound) / (upper_bound - lower_bound)
                delta2 = (upper_bound - y) / (upper_bound - lower_bound)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (upper_bound - lower_bound)
                y = np.clip(y, lower_bound, upper_bound)
                child_vars[i] = y
        
        child = Individual(decision_vars=child_vars)
        return child
    
    @staticmethod
    def gaussian_mutation(individual, bounds, sigma=0.1, prob_mutation=1.0):
        """
        高斯变异（Gaussian Mutation）
        
        对决策变量添加高斯噪声，噪声的标准差为sigma*(upper-lower)。
        
        参数:
            individual (Individual): 要变异的个体
            bounds (list): 决策变量的边界
            sigma (float): 噪声尺度，默认为0.1
            prob_mutation (float): 变异概率，默认为1.
            
        返回:
            Individual: 变异后的新个体
        """
        child_vars = individual.decision_vars.copy()
        
        for i in range(len(child_vars)):
            if np.random.random() <= prob_mutation / len(child_vars):
                lower_bound, upper_bound = bounds[i]
                mutation = np.random.normal(0, sigma * (upper_bound - lower_bound))
                child_vars[i] = np.clip(child_vars[i] + mutation, lower_bound, upper_bound)
        
        child = Individual(decision_vars=child_vars)
        return child
    
    @staticmethod
    def uniform_mutation(individual, bounds, prob_mutation=1.0):
        """
        均匀变异（Uniform Mutation）
        
        将决策变量随机设置为边界内的任意值。
        
        参数:
            individual (Individual): 要变异的个体
            bounds (list): 决策变量的边界
            prob_mutation (float): 变异概率，默认为1.0
            
        返回:
            Individual: 变异后的新个体
        """
        child_vars = individual.decision_vars.copy()
        
        for i in range(len(child_vars)):
            if np.random.random() <= prob_mutation / len(child_vars):
                lower_bound, upper_bound = bounds[i]
                child_vars[i] = np.random.uniform(lower_bound, upper_bound)
        
        child = Individual(decision_vars=child_vars)
        return child


class GeneticOperators:
    """
    遗传算子封装类
    
   
    
    封装交叉和变异算子，提供统一的接口。
    """
    
    def __init__(self, bounds, crossover_type='sbx', mutation_type='polynomial', 
                 eta_c=20, eta_m=20, prob_crossover=0.9, prob_mutation=1.0):
        """
        初始化遗传算子
        
        参数:
            bounds (list): 决策变量的边界
            crossover_type (str): 交叉算子类型，'sbx'、'uniform'或'arithmetic'
            mutation_type (str): 变异算子类型，'polynomial'、'gaussian'或'uniform'
            eta_c (float): 交叉分布指数
            eta_m (float): 变异分布指数
            prob_crossover (float): 交叉概率
            prob_mutation (float): 变异概率
        """
        self.bounds = bounds
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
    
    def crossover(self, parent1, parent2):
        """
        执行交叉操作
        
        参数:
            parent1 (Individual): 第一个父代个体
            parent2 (Individual): 第二个父代个体
            
        返回:
            tuple: (child1, child2) 两个子代个体
        """
        if self.crossover_type == 'sbx':
            return CrossoverOperator.sbx_crossover(
                parent1, parent2, self.bounds, self.eta_c, self.prob_crossover
            )
        elif self.crossover_type == 'uniform':
            return CrossoverOperator.uniform_crossover(
                parent1, parent2, self.prob_crossover
            )
        elif self.crossover_type == 'arithmetic':
            return CrossoverOperator.arithmetic_crossover(
                parent1, parent2, self.prob_crossover
            )
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")
    
    def mutate(self, individual):
        """
        执行变异操作
        
        参数:
            individual (Individual): 要变异的个体
            
        返回:
            Individual: 变异后的新个体
        """
        if self.mutation_type == 'polynomial':
            return MutationOperator.polynomial_mutation(
                individual, self.bounds, self.eta_m, self.prob_mutation
            )
        elif self.mutation_type == 'gaussian':
            return MutationOperator.gaussian_mutation(
                individual, self.bounds, prob_mutation=self.prob_mutation
            )
        elif self.mutation_type == 'uniform':
            return MutationOperator.uniform_mutation(
                individual, self.bounds, self.prob_mutation
            )
        else:
            raise ValueError(f"Unknown mutation type: {self.mutation_type}")
    
    def generate_offspring(self, population, n_offspring):
        """
        生成指定数量的后代
        
        参数:
            population (list): 父代种群
            n_offspring (int): 需要生成的后代数量
            
        返回:
            list: 后代个体列表
        """
        offspring = []
        
        while len(offspring) < n_offspring:
            parent1, parent2 = np.random.choice(population, 2, replace=False)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:n_offspring]
