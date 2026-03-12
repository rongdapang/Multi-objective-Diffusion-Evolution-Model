"""
自适应调度器模块 - 动态调整扩散模型和遗传算法的比例
"""

import numpy as np
from typing import Dict, Optional


class AdaptiveScheduler:
    """
    自适应调度器，动态调整扩散模型后代的生成比例
    """
    
    def __init__(self, base_dm_ratio: float = 0.4, 
                 min_ratio: float = 0.1, max_ratio: float = 0.7,
                 window_size: int = 10):
        """
        初始化调度器
        
        Args:
            base_dm_ratio: 基础扩散模型比例
            min_ratio: 最小比例
            max_ratio: 最大比例
            window_size: 性能历史窗口大小
        """
        self.base_dm_ratio = base_dm_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.window_size = window_size
        
        self.current_ratio = base_dm_ratio
        self.performance_history = []
        self.dm_survival_history = []
        self.generation = 0
        
    def get_dm_offspring_count(self, total_offspring: int, 
                                population_stats: Optional[Dict] = None) -> int:
        """
        获取扩散模型后代数量
        
        Args:
            total_offspring: 总后代数量
            population_stats: 种群统计信息
            
        Returns:
            扩散模型后代数量
        """
        # 根据性能历史调整比例
        self._update_ratio()
        
        # 计算扩散模型后代数量
        dm_count = int(total_offspring * self.current_ratio)
        
        return dm_count
    
    def _update_ratio(self):
        """
        根据性能历史更新扩散模型比例
        """
        self.generation += 1
        
        # 根据DM后代的存活率调整比例
        if len(self.dm_survival_history) >= self.window_size:
            recent_survival = np.mean(self.dm_survival_history[-self.window_size:])
            
            # 如果存活率高，增加比例；如果存活率低，减少比例
            if recent_survival > 0.3:
                self.current_ratio = min(self.current_ratio * 1.05, self.max_ratio)
            elif recent_survival < 0.1:
                self.current_ratio = max(self.current_ratio * 0.95, self.min_ratio)
        
        # 根据种群多样性调整（如果有统计信息）
        if len(self.performance_history) >= self.window_size:
            recent_improvement = self._calculate_improvement()
            
            if recent_improvement < 0.01:
                # 如果改进很小，增加探索（增加DM比例）
                self.current_ratio = min(self.current_ratio * 1.02, self.max_ratio)
    
    def _calculate_improvement(self) -> float:
        """
        计算最近几代的改进程度
        """
        if len(self.performance_history) < self.window_size * 2:
            return 1.0
        
        recent = np.mean(self.performance_history[-self.window_size:])
        previous = np.mean(self.performance_history[-2*self.window_size:-self.window_size])
        
        if previous == 0:
            return 1.0
        
        improvement = (previous - recent) / abs(previous)
        return max(0, improvement)
    
    def update_performance(self, population_stats: Dict):
        """
        更新种群性能历史
        
        Args:
            population_stats: 种群统计信息
        """
        if 'objs' in population_stats:
            objs = population_stats['objs']
            # 计算平均目标值
            avg_obj = np.mean(np.sum(objs, axis=1))
            self.performance_history.append(avg_obj)
            
            # 限制历史长度
            if len(self.performance_history) > self.window_size * 3:
                self.performance_history = self.performance_history[-self.window_size * 3:]
    
    def record_dm_performance(self, survival_rate: float):
        """
        记录扩散模型后代的存活率
        
        Args:
            survival_rate: 存活率（0-1之间）
        """
        self.dm_survival_history.append(survival_rate)
        
        # 限制历史长度
        if len(self.dm_survival_history) > self.window_size * 3:
            self.dm_survival_history = self.dm_survival_history[-self.window_size * 3:]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取调度器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'current_ratio': self.current_ratio,
            'avg_survival_rate': np.mean(self.dm_survival_history) if self.dm_survival_history else 0.0,
            'generation': self.generation
        }
