"""
遗传操作算子实现

包含模拟二进制交叉（SBX）和多项式变异算子。
"""

import numpy as np
from typing import List
from .core import Individual, Population


class OperatorGA:
    """遗传操作算子类"""
    
    def __init__(self, crossover_prob: float = 1.0, mutation_prob: Optional[float] = None,
                 eta_crossover: float = 20.0, eta_mutation: float = 20.0):
        """
        初始化遗传操作算子
        
        参数:
            crossover_prob: 交叉概率
            mutation_prob: 变异概率（None则使用 1/D）
            eta_crossover: 交叉分布指数
            eta_mutation: 变异分布指数
        """
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
        
    def evolve(self, mating_pool: List[Individual], problem) -> List[np.ndarray]:
        """
        对交配池进行进化操作
        
        参数:
            mating_pool: 交配池中的个体
            problem: 优化问题
            
        返回:
            生成的后代决策变量列表
        """
        n_individuals = len(mating_pool)
        n_variables = problem.n_variables
        
        # 设置变异概率
        if self.mutation_prob is None:
            mutation_prob = 1.0 / n_variables
        else:
            mutation_prob = self.mutation_prob
            
        offspring_decisions = []
        
        # 配对进行交叉和变异
        for i in range(0, n_individuals, 2):
            if i + 1 < n_individuals:
                # 有两个父代，可以进行交叉
                parent1 = mating_pool[i].decision_vars
                parent2 = mating_pool[i+1].decision_vars
                
                # 交叉
                if np.random.rand() < self.crossover_prob:
                    off1, off2 = self._sbx_crossover(parent1, parent2, 
                                                     problem.lower_bound, problem.upper_bound)
                else:
                    off1, off2 = parent1.copy(), parent2.copy()
                    
                # 变异
                off1 = self._polynomial_mutation(off1, problem.lower_bound, problem.upper_bound, 
                                                mutation_prob)
                off2 = self._polynomial_mutation(off2, problem.lower_bound, problem.upper_bound, 
                                                mutation_prob)
                
                offspring_decisions.append(off1)
                if len(offspring_decisions) < n_individuals:
                    offspring_decisions.append(off2)
            else:
                # 奇数个个体，直接变异
                parent = mating_pool[i].decision_vars
                offspring = self._polynomial_mutation(parent.copy(), 
                                                     problem.lower_bound, problem.upper_bound,
                                                     mutation_prob)
                offspring_decisions.append(offspring)
                
        return offspring_decisions
        
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                        lower_bound: np.ndarray, upper_bound: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟二进制交叉 (SBX)
        
        参数:
            parent1: 父代1
            parent2: 父代2
            lower_bound: 下界
            upper_bound: 上界
            
        返回:
            两个后代的决策变量
        """
        n_variables = len(parent1)
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        for i in range(n_variables):
            if np.random.rand() <= 0.5:  # 50%的概率进行交叉
                # 计算交叉参数
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                        
                    yl, yu = lower_bound[i], upper_bound[i]
                    
                    # 计算 beta
                    if y1 > yl:
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                    else:
                        beta = 1.0 + (2.0 * (y2 - yl) / (y2 - y1))
                        
                    alpha = 2.0 - beta ** (-(self.eta_crossover + 1.0))
                    
                    # 计算 beta_q
                    if alpha < 0:
                        beta_q = (2.0 * np.random.rand()) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        if np.random.rand() <= (1.0 / alpha):
                            beta_q = (2.0 * np.random.rand()) ** (1.0 / (self.eta_crossover + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - 2.0 * np.random.rand())) ** (1.0 / (self.eta_crossover + 1.0))
                            
                    # 生成后代
                    offspring1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    offspring2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    # 修复边界
                    offspring1[i] = np.clip(offspring1[i], lower_bound[i], upper_bound[i])
                    offspring2[i] = np.clip(offspring2[i], lower_bound[i], upper_bound[i])
                    
        return offspring1, offspring2
        
    def _polynomial_mutation(self, individual: np.ndarray, lower_bound: np.ndarray,
                            upper_bound: np.ndarray, mutation_prob: float) -> np.ndarray:
        """
        多项式变异
        
        参数:
            individual: 个体决策变量
            lower_bound: 下界
            upper_bound: 上界
            mutation_prob: 变异概率
            
        返回:
            变异后的决策变量
        """
        n_variables = len(individual)
        mutated = individual.copy()
        
        for i in range(n_variables):
            if np.random.rand() < mutation_prob:
                y = individual[i]
                yl, yu = lower_bound[i], upper_bound[i]
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                mut_pow = 1.0 / (self.eta_mutation + 1.0)
                
                if np.random.rand() <= 0.5:
                    # 左侧变异
                    xy = 1.0 - delta1
                    if xy < 0:
                        val = 0.0
                    else:
                        val = 2.0 * np.random.rand() + (1.0 - 2.0 * np.random.rand()) * (1.0 - xy) ** (self.eta_mutation + 1.0)
                        if val < 0:
                            delta_q = val ** mut_pow - 1.0
                        else:
                            delta_q = 1.0 - val ** mut_pow
                else:
                    # 右侧变异
                    xy = 1.0 - delta2
                    if xy < 0:
                        val = 0.0
                    else:
                        val = 2.0 * (1.0 - np.random.rand()) + 2.0 * (np.random.rand() - 0.5) * (1.0 - xy) ** (self.eta_mutation + 1.0)
                        if val < 0:
                            delta_q = val ** mut_pow - 1.0
                        else:
                            delta_q = 1.0 - val ** mut_pow
                            
                # 应用变异
                mutated[i] = y + delta_q * (yu - yl)
                
                # 修复边界
                mutated[i] = np.clip(mutated[i], yl, yu)
                
        return mutated