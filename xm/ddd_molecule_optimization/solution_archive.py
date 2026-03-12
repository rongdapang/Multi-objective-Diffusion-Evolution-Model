import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Optional

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
            if sol['smiles'] not in seen_smiles and sol['smiles'] is not None:
                seen_smiles.add(sol['smiles'])
                unique_solutions.append(sol)
        
        # 添加到存档
        self.solutions.extend(unique_solutions)
        
        # 如果超过最大容量，根据超体积贡献进行选择
        if len(self.solutions) > self.max_size:
            self._prune()
    
    def _prune(self):
        """
        根据超体积贡献剪枝存档
        """
        # 计算每个解的超体积贡献
        contributions = []
        objs = np.array([sol['objs'] for sol in self.solutions])
        
        for i in range(len(self.solutions)):
            # 移除第i个解后的超体积
            objs_without_i = np.delete(objs, i, axis=0)
            hv_without_i = self.calculate_hypervolume(objs_without_i, self.ref_point)
            # 原始超体积
            hv_original = self.calculate_hypervolume(objs, self.ref_point)
            # 贡献值
            contribution = hv_original - hv_without_i
            contributions.append(contribution)
        
        # 按照贡献值排序，保留贡献大的解
        sorted_indices = np.argsort(contributions)[::-1]
        self.solutions = [self.solutions[i] for i in sorted_indices[:self.max_size]]
    
    def get_target_objectives(self, n_targets: int) -> np.ndarray:
        """
        返回代表性的目标向量
        
        Args:
            n_targets: 目标向量数量
        
        Returns:
            目标向量矩阵
        """
        if len(self.solutions) == 0:
            # 随机生成目标向量
            return np.random.uniform([0.8, 1.5, 100], [1.0, 2.5, 300], (n_targets, 3))
        
        # 使用k-means聚类选择代表性目标
        objs = np.array([sol['objs'] for sol in self.solutions])
        
        if len(objs) < n_targets:
            # 如果解的数量少于目标数量，随机选择并补充
            selected_indices = np.random.choice(len(objs), len(objs), replace=False)
            selected_objs = objs[selected_indices]
            # 补充随机目标
            random_objs = np.random.uniform([0.8, 1.5, 100], [1.0, 2.5, 300], (n_targets - len(objs), 3))
            return np.vstack([selected_objs, random_objs])
        else:
            # 使用k-means聚类
            kmeans = KMeans(n_clusters=n_targets, random_state=42)
            kmeans.fit(objs)
            return kmeans.cluster_centers_
    
    def get_training_data(self) -> dict:
        """
        返回存档中的所有解，用于训练扩散模型
        
        Returns:
            包含z和objs的字典
        """
        if len(self.solutions) == 0:
            return {'z': np.array([]), 'objs': np.array([])}
        
        z = np.array([sol['z'] for sol in self.solutions])
        objs = np.array([sol['objs'] for sol in self.solutions])
        return {'z': z, 'objs': objs}
    
    def get_objectives(self) -> np.ndarray:
        """
        返回存档中所有解的目标值矩阵
        
        Returns:
            目标值矩阵
        """
        if len(self.solutions) == 0:
            return np.array([])
        
        return np.array([sol['objs'] for sol in self.solutions])
    
    def calculate_hypervolume(self, objs: np.ndarray, ref_point: np.ndarray) -> float:
        """
        计算超体积
        
        Args:
            objs: 目标值矩阵
            ref_point: 参考点
        
        Returns:
            超体积值
        """
        if len(objs) == 0:
            return 0.0
        
        # 简化的超体积计算（蒙特卡洛方法）
        n_points = 10000
        dim = objs.shape[1]
        
        # 生成随机点
        random_points = np.random.uniform(0, ref_point, (n_points, dim))
        
        # 计算被支配的点的数量
        dominated = 0
        for point in random_points:
            # 检查是否被至少一个解支配
            for obj in objs:
                if np.all(obj <= point):
                    dominated += 1
                    break
        
        # 计算超体积
        volume = dominated / n_points * np.prod(ref_point)
        return volume
