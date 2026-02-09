"""
多样性评估模块

该模块实现了多种多样性评估指标和条件编码器，包括：
- DiversityMetrics: 多样性指标计算类
- ConditionEncoder: 条件编码器
- AdaptiveDiversityThreshold: 自适应多样性阈值

多样性评估是DM-MOEA算法的关键组件，用于判断种群是否需要更多的探索。
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DiversityMetrics:
    """
    多样性指标计算类
    
    提供多种多样性评估方法，用于衡量种群在解空间中的分布情况。
    """
    
    @staticmethod
    def compute_crowding_distance_diversity(population):
        """
        基于拥挤距离的多样性
        
        计算种群中个体的平均拥挤距离（排除边界解）。
        拥挤距离越大，表示种群在目标空间中的分布越稀疏。
        
        参数:
            population (list): 个体列表
            
        返回:
            float: 平均拥挤距离
        """
        if len(population) == 0:
            return 0.0
        
        crowding_distances = [ind.crowding_distance for ind in population]
        mean_crowding = np.mean([cd for cd in crowding_distances if cd != float('inf')])
        return mean_crowding if not np.isnan(mean_crowding) else 0.0
    
    @staticmethod
    def compute_euclidean_distance_diversity(population, space='objective'):
        """
        基于欧氏距离的多样性
        
        计算种群中所有个体对之间的平均欧氏距离。
        可以在目标空间或决策空间中计算。
        
        参数:
            population (list): 个体列表
            space (str): 计算空间，'objective'或'decision'
            
        返回:
            float: 平均欧氏距离
        """
        if len(population) < 2:
            return 0.0
        
        if space == 'objective':
            points = np.array([ind.objectives for ind in population])
        elif space == 'decision':
            points = np.array([ind.decision_vars for ind in population])
        else:
            raise ValueError(f"Unknown space type: {space}")
        
        distances = pdist(points, metric='euclidean')
        mean_distance = np.mean(distances)
        
        return mean_distance
    
    @staticmethod
    def compute_pca_variance_diversity(population, space='objective'):
        """
        基于PCA方差的多样性
        
        使用主成分分析计算种群在主要方向上的方差。
        方差越大，表示种群在解空间中的分布越广。
        
        参数:
            population (list): 个体列表
            space (str): 计算空间，'objective'或'decision'
            
        返回:
            float: 平均解释方差
        """
        if len(population) < 2:
            return 0.0
        
        if space == 'objective':
            points = np.array([ind.objectives for ind in population])
        elif space == 'decision':
            points = np.array([ind.decision_vars for ind in population])
        else:
            raise ValueError(f"Unknown space type: {space}")
        
        n_components = min(points.shape[0] - 1, points.shape[1])
        if n_components < 1:
            return 0.0
        
        pca = PCA(n_components=n_components)
        pca.fit(points)
        
        return np.mean(pca.explained_variance_)
    
    @staticmethod
    def compute_spread_diversity(population, reference_front=None):
        """
        基于分布的多样性
        
        计算种群相对于参考前沿的分布情况。
        使用标准差与平均值的比值来衡量分布的均匀性。
        
        参数:
            population (list): 个体列表
            reference_front (list): 参考前沿，如果为None则使用种群本身
            
        返回:
            float: 分布多样性指标
        """
        if len(population) < 2:
            return 0.0
        
        points = np.array([ind.objectives for ind in population])
        
        if reference_front is not None:
            ref_points = np.array([ind.objectives for ind in reference_front])
        else:
            ref_points = points
        
        distances = []
        for point in points:
            min_dist = np.min(np.linalg.norm(ref_points - point, axis=1))
            distances.append(min_dist)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist < 1e-10:
            return 0.0
        
        spread = std_dist / mean_dist
        return spread
    
    @staticmethod
    def compute_entropy_diversity(population, n_bins=10, space='objective'):
        """
        基于熵的多样性
        
        计算种群在目标空间或决策空间中的熵。
        熵越大，表示种群的分布越均匀。
        
        参数:
            population (list): 个体列表
            n_bins (int): 直方图的箱数，默认为10
            space (str): 计算空间，'objective'或'decision'
            
        返回:
            float: 平均熵
        """
        if len(population) == 0:
            return 0.0
        
        if space == 'objective':
            points = np.array([ind.objectives for ind in population])
        elif space == 'decision':
            points = np.array([ind.decision_vars for ind in population])
        else:
            raise ValueError(f"Unknown space type: {space}")
        
        entropy = 0.0
        for dim in range(points.shape[1]):
            hist, _ = np.histogram(points[:, dim], bins=n_bins, density=True)
            hist = hist[hist > 0]
            entropy -= np.sum(hist * np.log(hist + 1e-10))
        
        return entropy / points.shape[1]
    
    @staticmethod
    def compute_combined_diversity(population, weights=None):
        """
        组合多样性指标
        
        将多种多样性指标加权组合成一个综合指标。
        
        参数:
            population (list): 个体列表
            weights (dict): 各指标的权重，默认为均匀权重
            
        返回:
            tuple: (combined_diversity, metrics) 综合多样性和各指标值
        """
        if weights is None:
            weights = {
                'crowding': 0.3,
                'euclidean': 0.3,
                'pca': 0.2,
                'entropy': 0.2
            }
        
        metrics = {
            'crowding': DiversityMetrics.compute_crowding_distance_diversity(population),
            'euclidean': DiversityMetrics.compute_euclidean_distance_diversity(population),
            'pca': DiversityMetrics.compute_pca_variance_diversity(population),
            'entropy': DiversityMetrics.compute_entropy_diversity(population)
        }
        
        for key in metrics:
            if metrics[key] < 0:
                metrics[key] = 0.0
        
        combined_diversity = sum(weights[key] * metrics[key] for key in weights)
        
        return combined_diversity, metrics


class ConditionEncoder:
    """
    条件编码器
    
    将进化算法的状态信息编码为条件向量，用于引导扩散模型生成。
    """
    
    def __init__(self, problem):
        """
        初始化条件编码器
        
        参数:
            problem (Problem): 优化问题对象
        """
        self.problem = problem
        self.pca = None
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, population, max_generations=100):
        """
        拟合编码器
        
        使用当前种群的数据训练编码器。
        
        参数:
            population (list): 个体列表
            max_generations (int): 最大代数，用于归一化
        """
        if len(population) < 2:
            return
        
        sample_conditions = []
        for gen in range(min(max_generations, len(population))):
            condition = self._extract_condition_features(population, gen, max_generations)
            sample_conditions.append(condition)
        
        if len(sample_conditions) > 0:
            condition_array = np.array(sample_conditions)
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(condition_array)
            
            n_components = min(scaled_features.shape[1], scaled_features.shape[0] - 1)
            if n_components > 0:
                self.pca = PCA(n_components=n_components)
                self.pca.fit(scaled_features)
            
            self.is_fitted = True
    
    def _extract_condition_features(self, population, generation, max_generations):
        """
        提取条件特征
        
        参数:
            population (list): 个体列表
            generation (int): 当前代数
            max_generations (int): 最大代数
            
        返回:
            list: 条件特征列表
        """
        if len(population) == 0:
            return [0.0]
        
        fronts = self.fast_non_dominated_sort(population)
        pareto_front = fronts[0] if len(fronts) > 0 else []
        
        diversity, metrics = DiversityMetrics.compute_combined_diversity(population)
        
        condition_features = []
        
        condition_features.append(len(pareto_front) / len(population))
        
        condition_features.append(generation / max_generations)
        
        condition_features.append(diversity)
        
        if len(pareto_front) > 0:
            pf_objectives = np.array([ind.objectives for ind in pareto_front])
            condition_features.extend(np.mean(pf_objectives, axis=0))
            condition_features.extend(np.std(pf_objectives, axis=0))
        else:
            condition_features.extend([0.0] * (2 * self.problem.n_obj))
        
        return condition_features
    
    def encode(self, population, generation, max_generations):
        """
        编码进化状态
        
        将种群状态、代数等信息编码为条件向量。
        
        参数:
            population (list): 个体列表
            generation (int): 当前代数
            max_generations (int): 最大代数
            
        返回:
            np.ndarray: 编码后的条件向量
        """
        if len(population) == 0:
            return np.zeros(1)
        
        fronts = self.fast_non_dominated_sort(population)
        pareto_front = fronts[0] if len(fronts) > 0 else []
        
        diversity, metrics = DiversityMetrics.compute_combined_diversity(population)
        
        condition_features = []
        
        condition_features.append(len(pareto_front) / len(population))
        
        condition_features.append(generation / max_generations)
        
        condition_features.append(diversity)
        
        if len(pareto_front) > 0:
            pf_objectives = np.array([ind.objectives for ind in pareto_front])
            condition_features.extend(np.mean(pf_objectives, axis=0))
            condition_features.extend(np.std(pf_objectives, axis=0))
        else:
            condition_features.extend([0.0] * (2 * self.problem.n_obj))
        
        if self.is_fitted and self.scaler is not None:
            condition_array = np.array(condition_features).reshape(1, -1)
            scaled_condition = self.scaler.transform(condition_array)
            
            if self.pca is not None:
                encoded = self.pca.transform(scaled_condition)[0]
            else:
                encoded = scaled_condition[0]
            
            return encoded
        
        return np.array(condition_features)
    
    fast_non_dominated_sort = None


def set_fast_non_dominated_sort_for_encoder(encoder, sort_function):
    """
    设置条件编码器的非支配排序函数
    
    参数:
        encoder (ConditionEncoder): 条件编码器实例
        sort_function: 非支配排序函数
    """
    encoder.fast_non_dominated_sort = sort_function


class AdaptiveDiversityThreshold:
    """
    自适应多样性阈值
    
    根据多样性变化情况动态调整多样性阈值。
    """
    
    def __init__(self, initial_threshold=0.5, min_threshold=0.1, max_threshold=0.9, 
                 decay_rate=0.95, adaptation_rate=0.1):
        """
        初始化自适应多样性阈值
        
        参数:
            initial_threshold (float): 初始阈值
            min_threshold (float): 最小阈值
            max_threshold (float): 最大阈值
            decay_rate (float): 衰减率
            adaptation_rate (float): 适应率
        """
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.decay_rate = decay_rate
        self.adaptation_rate = adaptation_rate
        self.history = []
    
    def update(self, diversity, diversity_improved):
        """
        更新阈值
        
        参数:
            diversity (float): 当前多样性值
            diversity_improved (bool): 多样性是否改善
        """
        self.history.append(diversity)
        
        if diversity_improved:
            self.current_threshold = min(
                self.max_threshold,
                self.current_threshold * (1 + self.adaptation_rate)
            )
        else:
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold * self.decay_rate
            )
    
    def get_threshold(self):
        """
        获取当前阈值
        
        返回:
            float: 当前阈值
        """
        return self.current_threshold
    
    def reset(self):
        """
        重置阈值
        """
        self.current_threshold = self.initial_threshold
        self.history = []
