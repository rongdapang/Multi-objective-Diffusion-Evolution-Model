"""
性能评估指标模块

该模块实现了多目标优化算法常用的性能评估指标，包括：
- Hypervolume: 超体积指标
- IGD: 反世代距离（Inverted Generational Distance）

这些指标用于评估算法找到的帕累托前沿的质量。
"""

import numpy as np


class Hypervolume:
    """
    超体积指标
    
    超体积衡量帕累托前沿在目标空间中覆盖的体积。
    超体积越大，表示帕累托前沿的质量越好。
    
    对于两目标问题，超体积可以高效计算。
    """
    
    def __init__(self, reference_point):
        """
        初始化超体积计算器
        
        参数:
            reference_point (list or np.ndarray): 参考点，通常为最差点的上界
        """
        self.reference_point = np.array(reference_point)
    
    def compute(self, objectives):
        """
        计算超体积
        
        参数:
            objectives (np.ndarray): 帕累托前沿的目标函数值，形状为(n, n_obj)
            
        返回:
            float: 超体积值
        """
        if len(objectives) == 0:
            return 0.0
        
        if objectives.shape[1] == 2:
            return self._compute_2d_hypervolume(objectives)
        else:
            return self._compute_nd_hypervolume(objectives)
    
    def _compute_2d_hypervolume(self, objectives):
        """
        计算两目标问题的超体积
        
        使用扫描线算法高效计算两目标超体积。
        
        参数:
            objectives (np.ndarray): 目标函数值，形状为(n, 2)
            
        返回:
            float: 超体积值
        """
        sorted_objectives = objectives[np.argsort(objectives[:, 0])]
        
        hv = 0.0
        for i in range(len(sorted_objectives)):
            f1, f2 = sorted_objectives[i]
            
            if i == len(sorted_objectives) - 1:
                next_f1 = self.reference_point[0]
            else:
                next_f1 = sorted_objectives[i + 1][0]
            
            width = min(next_f1, self.reference_point[0]) - f1
            height = self.reference_point[1] - f2
            
            if width > 0 and height > 0:
                hv += width * height
        
        return hv
    
    def _compute_nd_hypervolume(self, objectives):
        """
        计算多目标问题的超体积（近似）
        
        使用蒙特卡洛方法近似计算高维超体积。
        
        参数:
            objectives (np.ndarray): 目标函数值，形状为(n, n_obj)
            
        返回:
            float: 超体积值
        """
        n_samples = 10000
        n_obj = objectives.shape[1]
        
        bounds = []
        for i in range(n_obj):
            min_val = np.min(objectives[:, i])
            max_val = self.reference_point[i]
            bounds.append((min_val, max_val))
        
        random_points = np.random.rand(n_samples, n_obj)
        for i in range(n_obj):
            random_points[:, i] = random_points[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        
        dominated_count = 0
        for point in random_points:
            for obj in objectives:
                if np.all(obj <= point):
                    dominated_count += 1
                    break
        
        total_volume = 1.0
        for i in range(n_obj):
            total_volume *= (bounds[i][1] - bounds[i][0])
        
        return (dominated_count / n_samples) * total_volume


class IGD:
    """
    反世代距离（Inverted Generational Distance）
    
    IGD衡量算法找到的帕累托前沿到真实帕累托前沿的平均距离。
    IGD越小，表示算法找到的解越接近真实帕累托前沿。
    """
    
    def __init__(self, reference_front):
        """
        初始化IGD计算器
        
        参数:
            reference_front (np.ndarray): 真实帕累托前沿的目标函数值
        """
        self.reference_front = np.array(reference_front)
    
    def compute(self, objectives):
        """
        计算IGD
        
        参数:
            objectives (np.ndarray): 算法找到的帕累托前沿的目标函数值，形状为(n, n_obj)
            
        返回:
            float: IGD值
        """
        if len(objectives) == 0:
            return float('inf')
        
        if len(self.reference_front) == 0:
            return float('inf')
        
        distances = []
        for point in objectives:
            min_dist = np.min(np.linalg.norm(self.reference_front - point, axis=1))
            distances.append(min_dist)
        
        return np.mean(distances)
