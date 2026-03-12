"""
精英解存档模块 - 用于保存和管理进化过程中发现的优秀解
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class SolutionArchive:
    """
    精英存档类，用于保存进化过程中发现的精英解
    """
    
    def __init__(self, max_size: int, ref_point: np.ndarray = None):
        """
        初始化存档
        
        Args:
            max_size: 存档最大容量
            ref_point: 参考点，用于计算超体积
        """
        self.max_size = max_size
        self.ref_point = ref_point if ref_point is not None else np.array([1.1, 1.1, 1.1])
        self.solutions = []  # 每个元素为字典 {'z': z, 'objs': objs, 'smiles': smiles}
        
    def add(self, new_solutions: List[dict]) -> None:
        """
        添加新解到存档
        
        Args:
            new_solutions: 新解列表
        """
        # 去重
        unique_solutions = []
        seen_smiles = set()
        
        for sol in new_solutions:
            if sol['smiles'] is not None and sol['smiles'] not in seen_smiles:
                seen_smiles.add(sol['smiles'])
                unique_solutions.append(sol)
        
        # 添加到存档
        self.solutions.extend(unique_solutions)
        
        # 如果超过最大容量，根据拥挤度进行选择
        if len(self.solutions) > self.max_size:
            self._prune_by_crowding_distance()
    
    def _prune_by_crowding_distance(self):
        """
        根据拥挤度距离剪枝存档
        """
        if len(self.solutions) <= self.max_size:
            return
        
        # 提取目标值
        objs = np.array([sol['objs'] for sol in self.solutions])
        n_obj = objs.shape[1]
        
        # 计算拥挤度距离
        crowding_distances = np.zeros(len(self.solutions))
        
        for m in range(n_obj):
            # 按当前目标排序
            sorted_indices = np.argsort(objs[:, m])
            sorted_objs = objs[sorted_indices, m]
            
            # 边界点的拥挤度设为无穷大
            crowding_distances[sorted_indices[0]] = np.inf
            crowding_distances[sorted_indices[-1]] = np.inf
            
            # 计算中间点的拥挤度
            obj_range = sorted_objs[-1] - sorted_objs[0]
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    crowding_distances[sorted_indices[i]] += (
                        sorted_objs[i + 1] - sorted_objs[i - 1]
                    ) / obj_range
        
        # 选择拥挤度最大的解
        selected_indices = np.argsort(crowding_distances)[-self.max_size:]
        self.solutions = [self.solutions[i] for i in selected_indices]
    
    def get_training_data(self) -> Dict[str, np.ndarray]:
        """
        获取用于训练扩散模型的数据
        
        Returns:
            包含'z'和'objs'的字典
        """
        if len(self.solutions) == 0:
            return {'z': np.array([]), 'objs': np.array([])}
        
        z_list = [sol['z'] for sol in self.solutions]
        objs_list = [sol['objs'] for sol in self.solutions]
        
        return {
            'z': np.array(z_list),
            'objs': np.array(objs_list)
        }
    
    def get_target_objectives(self, n: int) -> np.ndarray:
        """
        获取用于扩散模型采样的目标向量
        
        Args:
            n: 需要的向量数量
            
        Returns:
            目标向量数组 (shape: [n, n_obj])
        """
        if len(self.solutions) == 0:
            # 如果没有解，返回随机目标
            return np.random.uniform([0.5, 0.0, 100], [1.0, 3.0, 400], (n, 3))
        
        # 提取目标值
        objs = np.array([sol['objs'] for sol in self.solutions])
        
        # 计算目标范围
        obj_min = objs.min(axis=0)
        obj_max = objs.max(axis=0)
        
        # 在优秀区域采样目标
        # 对于最小化问题，采样接近最小值的区域
        target_min = obj_min
        target_max = obj_min + 0.3 * (obj_max - obj_min)
        
        # 生成目标向量
        targets = np.random.uniform(target_min, target_max, (n, len(obj_min)))
        
        return targets
    
    def get_pareto_front(self) -> List[dict]:
        """
        获取帕累托前沿解
        
        Returns:
            帕累托前沿解列表
        """
        if len(self.solutions) == 0:
            return []
        
        # 提取目标值
        objs = np.array([sol['objs'] for sol in self.solutions])
        
        # 找出非支配解
        is_pareto = np.ones(len(self.solutions), dtype=bool)
        
        for i in range(len(self.solutions)):
            if is_pareto[i]:
                for j in range(len(self.solutions)):
                    if i != j and is_pareto[j]:
                        # 检查j是否支配i
                        if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                            is_pareto[i] = False
                            break
        
        pareto_solutions = [self.solutions[i] for i in range(len(self.solutions)) if is_pareto[i]]
        return pareto_solutions
    
    def calculate_hypervolume(self, objs: np.ndarray = None, ref_point: np.ndarray = None) -> float:
        """
        计算超体积指标
        
        Args:
            objs: 目标值数组，如果为None则使用存档中的所有解
            ref_point: 参考点
            
        Returns:
            超体积值
        """
        if objs is None:
            if len(self.solutions) == 0:
                return 0.0
            objs = np.array([sol['objs'] for sol in self.solutions])
        
        if ref_point is None:
            ref_point = self.ref_point
        
        if len(objs) == 0:
            return 0.0
        
        # 简单的超体积计算（适用于2-3维）
        try:
            from pymoo.indicators.hv import Hypervolume
            hv = Hypervolume(ref_point=ref_point)
            return hv(objs)
        except Exception:
            # 简化的超体积估计
            return self._approximate_hypervolume(objs, ref_point)
    
    def _approximate_hypervolume(self, objs: np.ndarray, ref_point: np.ndarray) -> float:
        """
        近似计算超体积
        """
        if len(objs) == 0:
            return 0.0
        
        # 使用蒙特卡洛方法估计
        n_samples = 10000
        
        # 确定采样范围
        min_vals = np.minimum(objs.min(axis=0), ref_point)
        max_vals = ref_point
        
        # 生成随机样本
        samples = np.random.uniform(min_vals, max_vals, (n_samples, len(ref_point)))
        
        # 计算被支配的样本数
        dominated_count = 0
        for sample in samples:
            for obj in objs:
                if np.all(obj <= sample):
                    dominated_count += 1
                    break
        
        # 估计超体积
        volume = np.prod(max_vals - min_vals)
        hv = volume * dominated_count / n_samples
        
        return hv
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取存档统计信息
        
        Returns:
            统计信息字典
        """
        if len(self.solutions) == 0:
            return {
                'size': 0,
                'hypervolume': 0.0,
                'avg_qed': 0.0,
                'avg_logp_error': 0.0,
                'avg_mw': 0.0
            }
        
        objs = np.array([sol['objs'] for sol in self.solutions])
        
        # objs[:, 0] = -QED (最大化QED)
        # objs[:, 1] = |logP - target| (最小化偏差)
        # objs[:, 2] = MW (最小化分子量)
        
        return {
            'size': len(self.solutions),
            'hypervolume': self.calculate_hypervolume(),
            'avg_qed': -np.mean(objs[:, 0]),
            'avg_logp_error': np.mean(objs[:, 1]),
            'avg_mw': np.mean(objs[:, 2])
        }
