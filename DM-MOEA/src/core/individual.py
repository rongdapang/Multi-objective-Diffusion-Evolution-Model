"""
个体类模块

该模块定义了Individual类，用于表示多目标优化问题中的一个解（个体）。
每个个体包含决策变量、目标函数值、约束条件等信息，以及用于非支配排序的属性。
"""

import numpy as np


class Individual:
    """
    个体类，表示多目标优化问题中的一个解
    
    属性:
        decision_vars (np.ndarray): 决策变量向量
        objectives (np.ndarray): 目标函数值向量
        constraints (np.ndarray): 约束条件值向量（如果存在）
        rank (int): 非支配排序的等级，0表示在第一帕累托前沿
        crowding_distance (float): 拥挤距离，用于维持种群多样性
        domination_count (int): 被其他个体支配的次数
        dominated_solutions (set): 该个体支配的其他个体集合
    """
    
    def __init__(self, decision_vars=None, objectives=None, constraints=None):
        """
        初始化个体
        
        参数:
            decision_vars (np.ndarray): 决策变量向量
            objectives (np.ndarray): 目标函数值向量
            constraints (np.ndarray): 约束条件值向量
        """
        self.decision_vars = decision_vars
        self.objectives = objectives
        self.constraints = constraints
        self.rank = 0
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_solutions = set()

    def evaluate(self, problem):
        """
        评估个体的目标函数值和约束条件
        
        参数:
            problem (Problem): 优化问题对象，提供evaluate_objectives和evaluate_constraints方法
            
        异常:
            ValueError: 如果决策变量未设置
        """
        if self.decision_vars is None:
            raise ValueError("Decision variables must be set before evaluation")
        
        self.objectives = problem.evaluate_objectives(self.decision_vars)
        
        if problem.has_constraints():
            self.constraints = problem.evaluate_constraints(self.decision_vars)
    
    def dominates(self, other):
        """
        判断当前个体是否支配另一个个体
        
        在多目标优化中，个体A支配个体B当且仅当：
        1. A在所有目标上都不劣于B
        2. A在至少一个目标上严格优于B
        
        参数:
            other (Individual): 另一个个体
            
        返回:
            bool: 如果当前个体支配另一个个体返回True，否则返回False
            
        异常:
            ValueError: 如果任一个体的目标函数未评估
        """
        if self.objectives is None or other.objectives is None:
            raise ValueError("Both individuals must have evaluated objectives")
        
        at_least_one_better = False
        for obj1, obj2 in zip(self.objectives, other.objectives):
            if obj1 > obj2:
                return False
            if obj1 < obj2:
                at_least_one_better = True
        return at_least_one_better
    
    def is_feasible(self):
        """
        判断个体是否可行
        
        如果没有约束条件，默认返回True。
        如果有约束条件，所有约束值必须小于等于0才认为可行。
        
        返回:
            bool: 如果个体可行返回True，否则返回False
        """
        if self.constraints is None:
            return True
        return all(c <= 0 for c in self.constraints)
    
    def copy(self):
        """
        创建个体的深拷贝
        
        返回:
            Individual: 个体的副本
        """
        new_ind = Individual(
            decision_vars=self.decision_vars.copy() if self.decision_vars is not None else None,
            objectives=self.objectives.copy() if self.objectives is not None else None,
            constraints=self.constraints.copy() if self.constraints is not None else None
        )
        new_ind.rank = self.rank
        new_ind.crowding_distance = self.crowding_distance
        new_ind.domination_count = self.domination_count
        new_ind.dominated_solutions = self.dominated_solutions.copy()
        return new_ind
    
    def __repr__(self):
        """
        返回个体的字符串表示
        
        返回:
            str: 个体的字符串表示
        """
        return f"Individual(decision_vars={self.decision_vars}, objectives={self.objectives}, rank={self.rank})"
