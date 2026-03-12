import numpy as np
from typing import Dict, List

class AdaptiveScheduler:
    """
    自适应调度器，用于动态调整扩散模型后代的产生比例
    """
    def __init__(self, base_ratio: float = 0.4, min_ratio: float = 0.15, max_ratio: float = 0.5):
        """
        初始化自适应调度器
        
        Args:
            base_ratio: 基础扩散模型后代比例
            min_ratio: 最小比例
            max_ratio: 最大比例
        """
        self.base_ratio = base_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.dm_history = []  # 记录每代DM后代存活率
        self.stagnation_count = 0
        self.last_best = np.inf
    
    def get_dm_offspring_count(self, total_offspring: int, dm_stats: Dict) -> int:
        """
        计算应该由扩散模型生成的后代数
        
        Args:
            total_offspring: 总后代数
            dm_stats: 扩散模型统计信息
        
        Returns:
            扩散模型后代数
        """
        # 根据近期存活率调整比例
        recent_performance = np.mean(self.dm_history[-3:]) if len(self.dm_history) >= 3 else self.base_ratio
        ratio = self.base_ratio
        
        # 根据性能调整比例
        if recent_performance > 0.3:
            ratio *= 1.2
        elif recent_performance < 0.1:
            ratio *= 0.8
        
        # 处理进化停滞
        if self.stagnation_count > 5:
            ratio *= 0.9
        
        # 限制比例范围
        ratio = np.clip(ratio, self.min_ratio, self.max_ratio)
        
        return int(total_offspring * ratio)
    
    def update_performance(self, population: Dict):
        """
        更新性能统计，检测进化停滞
        
        Args:
            population: 当前种群
        """
        if 'objs' not in population or len(population['objs']) == 0:
            return
        
        # 假设第一个目标是最小化目标
        current_best = np.min(population['objs'][:, 0])
        
        if current_best >= self.last_best:
            # 没有改进，增加停滞计数
            self.stagnation_count += 1
        else:
            # 有改进，重置停滞计数
            self.stagnation_count = 0
        
        # 更新最佳值
        self.last_best = min(self.last_best, current_best)
    
    def record_dm_performance(self, survival_rate: float):
        """
        记录扩散模型后代的存活率
        
        Args:
            survival_rate: 存活率
        """
        self.dm_history.append(survival_rate)
        # 保持历史记录长度不超过10
        if len(self.dm_history) > 10:
            self.dm_history.pop(0)
