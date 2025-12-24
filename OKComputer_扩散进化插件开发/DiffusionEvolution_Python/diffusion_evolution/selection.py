"""
选择机制实现

包含环境选择和锦标赛选择，基于NSGA-II框架。
"""

import numpy as np
from typing import List
from .core import Individual, Population


class EnvironmentalSelection:
    """环境选择 - 基于非支配排序和拥挤距离"""
    
    def select(self, population: Population, n_select: int) -> List[int]:
        """
        选择个体
        
        参数:
            population: 种群
            n_select: 要选择个体数量
            
        返回:
            被选中个体的索引列表
        """
        if len(population) <= n_select:
            return list(range(len(population)))
            
        # 获取目标函数矩阵
        objectives = population.get_objective_matrix()
        n_objectives = objectives.shape[1]
        
        # 非支配排序
        fronts = self._non_dominated_sort(objectives)
        
        # 计算拥挤距离
        crowding_distances = self._calculate_crowding_distance(objectives, fronts)
        
        # 环境选择
        selected_indices = []
        current_front = 1
        
        while len(selected_indices) < n_select:
            # 获取当前前沿的个体
            current_front_indices = [i for i, front in enumerate(fronts) if front == current_front]
            
            if len(selected_indices) + len(current_front_indices) <= n_select:
                # 添加整个前沿
                selected_indices.extend(current_front_indices)
            else:
                # 需要选择部分个体，使用拥挤距离
                remaining_slots = n_select - len(selected_indices)
                
                # 按拥挤距离排序（降序）
                front_distances = [(i, crowding_distances[i]) for i in current_front_indices]
                front_distances.sort(key=lambda x: x[1], reverse=True)
                
                # 选择拥挤距离最大的个体
                for i in range(remaining_slots):
                    selected_indices.append(front_distances[i][0])
                    
            current_front += 1
            
        return selected_indices
        
    def _non_dominated_sort(self, objectives: np.ndarray) -> List[int]:
        """
        非支配排序
        
        参数:
            objectives: 目标函数矩阵 (n_individuals, n_objectives)
            
        返回:
            每个个体的前沿编号列表
        """
        n_individuals = objectives.shape[0]
        fronts = [0] * n_individuals
        
        # 记录每个个体被支配的次数
        dominated_counts = [0] * n_individuals
        
        # 记录每个个体支配的其他个体
        dominated_sets = [[] for _ in range(n_individuals)]
        
        # 比较所有个体对
        for i in range(n_individuals):
            for j in range(i + 1, n_individuals):
                # 检查支配关系
                if self._dominates(objectives[i], objectives[j]):
                    dominated_sets[i].append(j)
                    dominated_counts[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_sets[j].append(i)
                    dominated_counts[i] += 1
                    
        # 分配前沿编号
        current_front = 1
        remaining = list(range(n_individuals))
        
        while remaining:
            # 找到当前前沿的个体（没有被支配的）
            current_front_members = [i for i in remaining if dominated_counts[i] == 0]
            
            if not current_front_members:
                break
                
            # 分配前沿编号
            for i in current_front_members:
                fronts[i] = current_front
                
            # 更新支配计数
            for i in current_front_members:
                for j in dominated_sets[i]:
                    dominated_counts[j] -= 1
                    
            # 移除已分配前沿的个体
            remaining = [i for i in remaining if fronts[i] == 0]
            current_front += 1
            
        return fronts
        
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断 obj1 是否支配 obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
        
    def _calculate_crowding_distance(self, objectives: np.ndarray, fronts: List[int]) -> List[float]:
        """
        计算拥挤距离
        
        参数:
            objectives: 目标函数矩阵
            fronts: 前沿编号列表
            
        返回:
            拥挤距离列表
        """
        n_individuals = objectives.shape[0]
        n_objectives = objectives.shape[1]
        
        # 初始化拥挤距离
        crowding_distances = [0.0] * n_individuals
        
        # 为每个前沿计算拥挤距离
        max_front = max(fronts)
        
        for front in range(1, max_front + 1):
            front_indices = [i for i, f in enumerate(fronts) if f == front]
            
            if len(front_indices) <= 2:
                # 边界个体有无限拥挤距离
                for i in front_indices:
                    crowding_distances[i] = float('inf')
            else:
                # 计算每个目标的拥挤距离
                for m in range(n_objectives):
                    # 按当前目标排序
                    sorted_indices = sorted(front_indices, key=lambda i: objectives[i, m])
                    
                    # 边界个体有无限拥挤距离
                    crowding_distances[sorted_indices[0]] = float('inf')
                    crowding_distances[sorted_indices[-1]] = float('inf')
                    
                    # 计算内部个体的拥挤距离
                    obj_min = objectives[sorted_indices[0], m]
                    obj_max = objectives[sorted_indices[-1], m]
                    
                    if obj_max > obj_min:
                        for i in range(1, len(sorted_indices) - 1):
                            prev_obj = objectives[sorted_indices[i-1], m]
                            next_obj = objectives[sorted_indices[i+1], m]
                            crowding_distances[sorted_indices[i]] += (next_obj - prev_obj) / (obj_max - obj_min)
                            
        return crowding_distances


class TournamentSelection:
    """锦标赛选择"""
    
    def select(self, population: Population, n_select: int, tournament_size: int = 2) -> List[int]:
        """
        选择个体
        
        参数:
            population: 种群
            n_select: 要选择个体数量
            tournament_size: 锦标赛大小
            
        返回:
            被选中个体的索引列表
        """
        n_individuals = len(population)
        selected_indices = []
        
        # 获取目标函数和前沿信息
        objectives = population.get_objective_matrix()
        fronts = self._get_fronts(population)
        
        for _ in range(n_select):
            # 随机选择锦标赛参与者
            tournament_indices = np.random.choice(n_individuals, tournament_size, replace=False)
            
            # 找到最好的个体
            best_idx = tournament_indices[0]
            best_front = fronts[best_idx]
            
            for idx in tournament_indices[1:]:
                current_front = fronts[idx]
                
                # 比较前沿等级
                if current_front < best_front:
                    best_idx = idx
                    best_front = current_front
                elif current_front == best_front:
                    # 相同前沿，比较拥挤距离
                    # 这里简化处理，选择第一个
                    pass
                    
            selected_indices.append(best_idx)
            
        return selected_indices
        
    def _get_fronts(self, population: Population) -> List[int]:
        """获取个体的前沿编号"""
        objectives = population.get_objective_matrix()
        
        # 简化的非支配排序
        n_individuals = len(population)
        fronts = [1] * n_individuals
        
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i != j:
                    # 检查是否被支配
                    if self._dominates(objectives[j], objectives[i]):
                        fronts[i] += 1
                        
        return fronts
        
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断支配关系"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)