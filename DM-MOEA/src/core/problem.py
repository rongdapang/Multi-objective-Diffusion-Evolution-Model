"""
问题定义模块

该模块定义了多目标优化问题的抽象基类Problem，以及多个标准的测试问题（ZDT和DTLZ系列）。
这些测试问题广泛用于评估多目标进化算法的性能。
"""

import numpy as np
from abc import ABC, abstractmethod


class Problem(ABC):
    """
    多目标优化问题抽象基类
    
    所有具体的问题类都应该继承此类并实现evaluate_objectives方法。
    
    属性:
        n_vars (int): 决策变量的数量
        n_obj (int): 目标函数的数量
        n_constraints (int): 约束条件的数量
        bounds (list): 每个决策变量的上下界，格式为[(lower1, upper1), ...]
    """
    
    def __init__(self, n_vars, n_obj, n_constraints=0, bounds=None):
        """
        初始化问题
        
        参数:
            n_vars (int): 决策变量的数量
            n_obj (int): 目标函数的数量
            n_constraints (int): 约束条件的数量，默认为0
            bounds (list): 每个决策变量的上下界，如果为None则默认为[0, 1]
        """
        self.n_vars = n_vars
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.bounds = bounds if bounds is not None else [(0.0, 1.0) for _ in range(n_vars)]
    
    @abstractmethod
    def evaluate_objectives(self, x):
        """
        评估目标函数（抽象方法，子类必须实现）
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            np.ndarray: 目标函数值向量
        """
        pass
    
    def evaluate_constraints(self, x):
        """
        评估约束条件（默认实现为无约束）
        
        子类可以重写此方法以实现具体的约束条件。
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            np.ndarray: 约束条件值向量，约束满足条件为值<=0
        """
        return np.zeros(self.n_constraints)
    
    def has_constraints(self):
        """
        判断问题是否有约束条件
        
        返回:
            bool: 如果有约束条件返回True，否则返回False
        """
        return self.n_constraints > 0
    
    def is_valid(self, x):
        """
        判断决策变量是否在有效范围内
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            bool: 如果决策变量在边界内返回True，否则返回False
        """
        for i, (lower, upper) in enumerate(self.bounds):
            if x[i] < lower or x[i] > upper:
                return False
        return True


class ZDT1(Problem):
    """
    ZDT1测试问题
    
    ZDT1是一个具有凸帕累托前沿的两目标测试问题。
    帕累托前沿的解析解为：f2 = 1 - sqrt(f1)，其中f1 ∈ [0, 1]
    
    参考文献:
        Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective 
        evolutionary algorithms: Empirical results. Evolutionary computation, 8(2), 173-195.
    """
    
    def __init__(self, n_vars=30):
        """
        初始化ZDT1问题
        
        参数:
            n_vars (int): 决策变量数量，默认为30
        """
        super().__init__(n_vars=n_vars, n_obj=2, bounds=[(0.0, 1.0) for _ in range(n_vars)])
        self.reference_point = [1.1, 1.1]
    
    def evaluate_objectives(self, x):
        """
        评估ZDT1问题的目标函数
        
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (n - 1)
        h = 1 - sqrt(f1 / g)
        f2 = g * h
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            np.ndarray: 目标函数值向量 [f1, f2]
        """
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_vars - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])


class ZDT2(Problem):
    """
    ZDT2测试问题
    
    ZDT2是一个具有非凸帕累托前沿的两目标测试问题。
    帕累托前沿的解析解为：f2 = 1 - f1^2，其中f1 ∈ [0, 1]
    
    参考文献:
        Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective 
        evolutionary algorithms: Empirical results. Evolutionary computation, 8(2), 173-195.
    """
    
    def __init__(self, n_vars=30):
        """
        初始化ZDT2问题
        
        参数:
            n_vars (int): 决策变量数量，默认为30
        """
        super().__init__(n_vars=n_vars, n_obj=2, bounds=[(0.0, 1.0) for _ in range(n_vars)])
        self.reference_point = [1.1, 1.1]
    
    def evaluate_objectives(self, x):
        """
        评估ZDT2问题的目标函数
        
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (n - 1)
        h = 1 - (f1 / g)^2
        f2 = g * h
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            np.ndarray: 目标函数值向量 [f1, f2]
        """
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_vars - 1)
        h = 1 - (f1 / g) ** 2
        f2 = g * h
        return np.array([f1, f2])


class ZDT3(Problem):
    """
    ZDT3测试问题
    
    ZDT3是一个具有不连续帕累托前沿的两目标测试问题。
    帕累托前沿由多个不连续的曲线段组成，这测试算法维持多样性的能力。
    
    参考文献:
        Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective 
        evolutionary algorithms: Empirical results. Evolutionary computation, 8(2), 173-195.
    """
    
    def __init__(self, n_vars=30):
        """
        初始化ZDT3问题
        
        参数:
            n_vars (int): 决策变量数量，默认为30
        """
        super().__init__(n_vars=n_vars, n_obj=2, bounds=[(0.0, 1.0) for _ in range(n_vars)])
        self.reference_point = [1.1, 1.1]
    
    def evaluate_objectives(self, x):
        """
        评估ZDT3问题的目标函数
        
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (n - 1)
        h = 1 - sqrt(f1 / g) - (f1 / g) * sin(10 * pi * f1)
        f2 = g * h
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            np.ndarray: 目标函数值向量 [f1, f2]
        """
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_vars - 1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])


class DTLZ1(Problem):
    """
    DTLZ1测试问题
    
    DTLZ1是一个可扩展到任意数量目标的多目标测试问题。
    帕累托前沿是一个线性超平面，该问题具有11^(k)-1个局部帕累托前沿，
    其中k = n_vars - n_obj + 1，这使得算法很容易陷入局部最优。
    
    参考文献:
        Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2002). Scalable 
        multi-objective optimization test problems. In Evolutionary multi-criterion 
        optimization (pp. 82-104). Springer, Berlin, Heidelberg.
    """
    
    def __init__(self, n_vars=7, n_obj=3):
        """
        初始化DTLZ1问题
        
        参数:
            n_vars (int): 决策变量数量，默认为7
            n_obj (int): 目标函数数量，默认为3
        """
        super().__init__(n_vars=n_vars, n_obj=n_obj, bounds=[(0.0, 1.0) for _ in range(n_vars)])
    
    def evaluate_objectives(self, x):
        """
        评估DTLZ1问题的目标函数
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            np.ndarray: 目标函数值向量
        """
        k = self.n_vars - self.n_obj + 1
        g = 100 * (k + np.sum((x[-k:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[-k:] - 0.5))))
        
        f = []
        for i in range(self.n_obj):
            f_val = 0.5 * (1 + g)
            for j in range(self.n_obj - i - 1):
                f_val *= x[j]
            if i > 0:
                f_val *= 1 - x[self.n_obj - i - 1]
            f.append(f_val)
        
        return np.array(f)


class DTLZ2(Problem):
    """
    DTLZ2测试问题
    
    DTLZ2是一个可扩展到任意数量目标的多目标测试问题。
    帕累托前沿是单位球面的一部分，该问题用于测试算法处理球形帕累托前沿的能力。
    
    参考文献:
        Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2002). Scalable 
        multi-objective optimization test problems. In Evolutionary multi-criterion 
        optimization (pp. 82-104). Springer, Berlin, Heidelberg.
    """
    
    def __init__(self, n_vars=12, n_obj=3):
        """
        初始化DTLZ2问题
        
        参数:
            n_vars (int): 决策变量数量，默认为12
            n_obj (int): 目标函数数量，默认为3
        """
        super().__init__(n_vars=n_vars, n_obj=n_obj, bounds=[(0.0, 1.0) for _ in range(n_vars)])
    
    def evaluate_objectives(self, x):
        """
        评估DTLZ2问题的目标函数
        
        参数:
            x (np.ndarray): 决策变量向量
            
        返回:
            np.ndarray: 目标函数值向量
        """
        k = self.n_vars - self.n_obj + 1
        g = np.sum((x[-k:] - 0.5) ** 2)
        
        f = []
        for i in range(self.n_obj):
            f_val = 1 + g
            for j in range(self.n_obj - i - 1):
                f_val *= np.cos(x[j] * np.pi / 2)
            if i > 0:
                f_val *= np.sin(x[self.n_obj - i - 1] * np.pi / 2)
            f.append(f_val)
        
        return np.array(f)
