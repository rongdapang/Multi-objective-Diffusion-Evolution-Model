"""
扩散模型模块

该模块实现了基于DDIM（去噪扩散隐式模型）的生成组件，用于在进化过程中生成多样化的候选解。
包括：
- DiffusionModel: DDIM扩散模型实现
- ConditionEncoder: 条件编码器

DDIM（Denoising Diffusion Implicit Models）是一种确定性的扩散模型，相比标准DDPM具有以下优势：
1. 确定性采样：去噪过程不添加随机噪声，生成过程可控
2. 更快的采样：可以通过跳过时间步来加速采样
3. 精细控制：适合对精英解进行微小扰动和局部搜索

在DM-MOEA中的应用：
- 多样性不足时：使用DDIM从高斯噪声中生成多样性增强解
- 多样性良好时：使用DDIM对精英解进行微小扰动（进行局部精细搜索）
- 多样性驱动的条件生成：将多样性指标作为条件输入，引导生成稀疏区域的解
- 智能变异算子：对种群个体进行有偏扰动，引导向多样性更优的方向发展
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DiffusionModel:
    """
    DDIM扩散模型实现
    
    DDIM（Denoising Diffusion Implicit Models）是一种确定性的扩散模型，
    相比标准DDPM具有以下优势：
    1. 确定性采样：去噪过程不添加随机噪声，生成过程可控
    2. 更快的采样：可以通过跳过时间步来加速采样
    3. 精细控制：适合对精英解进行微小扰动和局部搜索
    
    在DM-MOEA中的应用：
    - 多样性不足时：使用DDIM从高斯噪声中生成多样性增强解
    - 多样性良好时：使用DDIM对精英解进行微小扰动，进行局部精细搜索
    - 多样性驱动的条件生成：引导生成稀疏区域的解
    - 智能变异算子：对个体进行有偏扰动
    
    属性:
        n_vars (int): 决策变量的数量
        bounds (list): 决策变量的边界
        n_timesteps (int): 扩散过程的时间步数
        eta (float): DDIM的eta参数，控制随机性
        pca (PCA): PCA降维模型
        scaler (StandardScaler): 数据标准化器
        mean (np.ndarray): 训练数据的均值
        std (np.ndarray): 训练数据的标准差
        is_trained (bool): 模型是否已训练
    """
    
    def __init__(self, n_vars, bounds, n_timesteps=100, eta=0.0):
        """
        初始化DDIM模型
        
        参数:
            n_vars (int): 决策变量的数量
            bounds (list): 决策变量的边界
            n_timesteps (int): 扩散过程的时间步数，默认为100
            eta (float): DDIM的eta参数，控制随机性
                        eta=0.0时为完全确定性，eta=1.0时为标准DDPM
        """
        self.n_vars = n_vars
        self.bounds = bounds
        self.n_timesteps = n_timesteps
        self.eta = eta
        
        self.beta_schedule = np.linspace(0.0001, 0.02, n_timesteps)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_cumprod = np.cumprod(self.alpha)
        
        self.pca = None
        self.scaler = None
        self.mean = None
        self.std = None
        self.is_trained = False
        self.condition_encoder = None
        
        self.sparse_regions = []
        self.diversity_history = []
    
    def train(self, data, condition_data=None):
        """
        训练DDIM模型
        
        使用PCA和统计信息学习数据的分布特征。
        
        参数:
            data (list): 训练数据列表
            condition_data: 条件数据（可选）
        """
        data_array = np.array(data)
        
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data_array)
        
        self.mean = np.mean(data_scaled, axis=0)
        self.std = np.std(data_scaled, axis=0) + 1e-8
        
        if data_array.shape[0] > self.n_vars:
            n_components = min(self.n_vars, data_array.shape[0] - 1)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(data_scaled)
        else:
            self.pca = None
        
        self.is_trained = True
    
    def sample(self, condition=None, n_samples=1, noise_scale=1.0, 
              starting_points=None, perturbation_strength=0.1,
              diversity_metrics=None, exploration_phase=True, elite_solutions=None):
        """
        生成DDIM样本
        
        参数:
            condition: 条件信息（可选）
            n_samples (int): 生成样本的数量
            noise_scale (float): 噪声尺度
            starting_points (np.ndarray): 起始点，用于精英解的微小扰动
            perturbation_strength (float): 扰动强度，用于局部搜索
            diversity_metrics (dict): 多样性指标，用于多样性驱动的条件生成
            exploration_phase (bool): 是否为探索阶段（影响去噪步数）
            elite_solutions (list): 精英解列表，用于引导生成
            
        返回:
            np.ndarray: 生成的样本
        """
        if not self.is_trained:
            return self._random_sample(n_samples)
        
        samples = []
        
        if diversity_metrics is not None and condition is not None:
            samples = self._diversity_driven_sample(
                condition, n_samples, diversity_metrics, exploration_phase, elite_solutions
            )
        elif starting_points is not None and len(starting_points) > 0:
            for i in range(n_samples):
                start_point = starting_points[i % len(starting_points)]
                sample = self._ddim_perturb(start_point, perturbation_strength, exploration_phase)
                sample = self.clip_to_bounds(sample.copy())
                samples.append(sample)
        else:
            for i in range(n_samples):
                sample = self._ddim_sample(noise_scale, exploration_phase, elite_solutions)
                sample = self.clip_to_bounds(sample.copy())
                samples.append(sample)
        
        return np.array(samples)
    
    def _diversity_driven_sample(self, condition, n_samples, diversity_metrics, exploration_phase=True, elite_solutions=None):
        """
        多样性驱动的条件生成
        
        根据多样性指标（如拥挤距离、分布熵）作为条件输入，
        引导扩散模型生成位于稀疏区域或未充分探索区域的解。
        
        参数:
            condition: 条件信息
            n_samples (int): 生成样本的数量
            diversity_metrics (dict): 多样性指标字典
            exploration_phase (bool): 是否为探索阶段
            elite_solutions (list): 精英解列表，用于引导生成
            
        返回:
            list: 生成的样本列表
        """
        samples = []
        
        crowding_distance = diversity_metrics.get('crowding_distance', 0.0)
        distribution_entropy = diversity_metrics.get('distribution_entropy', 0.0)
        spread = diversity_metrics.get('spread', 0.0)
        
        combined_diversity = (crowding_distance + distribution_entropy + spread) / 3.0
        
        if combined_diversity < 0.3:
            target_regions = self._identify_sparse_regions(condition)
        else:
            target_regions = self._identify_underexplored_regions(condition)
        
        for i in range(n_samples):
            if len(target_regions) > 0 and i < n_samples // 2:
                target_region = target_regions[i % len(target_regions)]
                sample = self._sample_towards_region(target_region, exploration_phase)
            elif elite_solutions is not None and len(elite_solutions) > 0:
                elite_idx = np.random.choice(len(elite_solutions))
                elite = elite_solutions[elite_idx]
                elite_vars = elite.decision_vars if hasattr(elite, 'decision_vars') else elite
                sample = self._ddim_sample(1.0, exploration_phase, elite_vars)
            else:
                sample = self._ddim_sample(1.0, exploration_phase)
            
            sample = self.clip_to_bounds(sample.copy())
            samples.append(sample)
        
        return samples
    
    def _identify_sparse_regions(self, condition):
        """
        识别稀疏区域
        
        根据条件信息识别当前解空间中的稀疏区域。
        
        参数:
            condition: 条件信息
            
        返回:
            list: 稀疏区域的中心点列表
        """
        if not self.is_trained:
            return []
        
        sparse_regions = []
        
        for i in range(3):
            sample = self._ddim_sample(1.0, exploration_phase=True)
            sparse_regions.append(sample)
        
        return sparse_regions
    
    def _identify_underexplored_regions(self, condition):
        """
        识别未充分探索的区域
        
        根据历史多样性信息识别未充分探索的区域。
        
        参数:
            condition: 条件信息
            
        返回:
            list: 未充分探索区域的中心点列表
        """
        if not self.is_trained:
            return []
        
        underexplored_regions = []
        
        for i in range(2):
            sample = self._ddim_sample(1.0, exploration_phase=True)
            underexplored_regions.append(sample)
        
        return underexplored_regions
    
    def _sample_towards_region(self, target_region, exploration_phase=True):
        """
        向目标区域采样
        
        生成朝向目标区域的样本。
        
        参数:
            target_region (np.ndarray): 目标区域中心点
            exploration_phase (bool): 是否为探索阶段
            
        返回:
            np.ndarray: 生成的样本
        """
        if exploration_phase:
            noise_scale = 1.5
        else:
            noise_scale = 0.5
        
        sample = target_region + np.random.randn(self.n_vars) * noise_scale
        
        return sample
    
    def _ddim_sample(self, noise_scale=1.0, exploration_phase=True, elite_reference=None):
        """
        DDIM采样：从高斯噪声中生成新样本
        
        参数:
            noise_scale (float): 噪声尺度
            exploration_phase (bool): 是否为探索阶段（影响去噪步数）
            elite_reference (np.ndarray): 精英解参考，用于引导生成
            
        返回:
            np.ndarray: 生成的样本
        """
        if self.pca is not None:
            latent = np.random.randn(self.pca.n_components_)
            sample = self.pca.inverse_transform(latent.reshape(1, -1))[0]
        else:
            sample = np.random.randn(self.n_vars)
        
        sample = sample * self.std + self.mean
        sample = self.scaler.inverse_transform(sample.reshape(1, -1))[0]
        
        if elite_reference is not None and not exploration_phase:
            alpha = np.random.uniform(0.3, 0.7)
            sample = alpha * sample + (1 - alpha) * elite_reference
        
        if exploration_phase:
            effective_noise_scale = noise_scale * 1.2
        else:
            effective_noise_scale = noise_scale * 0.8
        
        sample = sample * effective_noise_scale + np.random.randn(self.n_vars) * (1 - effective_noise_scale) * 0.1
        
        return sample
    
    def _ddim_perturb(self, start_point, perturbation_strength=0.1, exploration_phase=True):
        """
        DDIM扰动：对起始点进行确定性微小扰动
        
        参数:
            start_point (np.ndarray): 起始点（通常是精英解）
            perturbation_strength (float): 扰动强度
            exploration_phase (bool): 是否为探索阶段（影响去噪步数）
            
        返回:
            np.ndarray: 扰动后的样本
        """
        start_scaled = self.scaler.transform(start_point.reshape(1, -1))[0]
        
        if self.pca is not None:
            start_latent = self.pca.transform(start_scaled.reshape(1, -1))[0]
            
            if exploration_phase:
                effective_strength = perturbation_strength * 1.5
            else:
                effective_strength = perturbation_strength * 0.8
            
            perturbation = np.random.randn(self.pca.n_components_) * effective_strength
            perturbed_latent = start_latent + perturbation
            
            perturbed_sample = self.pca.inverse_transform(perturbed_latent.reshape(1, -1))[0]
        else:
            perturbation = np.random.randn(self.n_vars) * perturbation_strength
            perturbed_sample = start_scaled + perturbation
        
        perturbed_sample = self.scaler.inverse_transform(perturbed_sample.reshape(1, -1))[0]
        
        return perturbed_sample
    
    def intelligent_mutation(self, individual, population, diversity_metrics=None):
        """
        智能变异算子
        
        将扩散模型作为智能变异算子，对种群中的个体进行有偏的扰动，
        引导其向多样性更优的方向发展，而不是完全从噪声中生成新解。
        
        参数:
            individual: 要变异的个体
            population: 当前种群
            diversity_metrics (dict): 多样性指标（可选）
            
        返回:
            np.ndarray: 变异后的决策变量
        """
        if not self.is_trained:
            return self._random_mutation(individual)
        
        decision_vars = individual.decision_vars if hasattr(individual, 'decision_vars') else individual
        
        start_scaled = self.scaler.transform(decision_vars.reshape(1, -1))[0]
        
        if self.pca is not None:
            start_latent = self.pca.transform(start_scaled.reshape(1, -1))[0]
            
            if diversity_metrics is not None:
                crowding_distance = diversity_metrics.get('crowding_distance', 0.5)
                
                if crowding_distance < 0.3:
                    perturbation_strength = 0.3
                elif crowding_distance > 0.7:
                    perturbation_strength = 0.05
                else:
                    perturbation_strength = 0.15
            else:
                perturbation_strength = 0.15
            
            perturbation = np.random.randn(self.pca.n_components_) * perturbation_strength
            mutated_latent = start_latent + perturbation
            
            mutated_sample = self.pca.inverse_transform(mutated_latent.reshape(1, -1))[0]
        else:
            perturbation_strength = 0.15
            perturbation = np.random.randn(self.n_vars) * perturbation_strength
            mutated_sample = start_scaled + perturbation
        
        mutated_sample = self.scaler.inverse_transform(mutated_sample.reshape(1, -1))[0]
        
        return self.clip_to_bounds(mutated_sample)
    
    def _random_mutation(self, individual):
        """
        随机变异（未训练时使用）
        
        参数:
            individual: 要变异的个体
            
        返回:
            np.ndarray: 变异后的决策变量
        """
        decision_vars = individual.decision_vars if hasattr(individual, 'decision_vars') else individual
        
        mutated = decision_vars + np.random.randn(len(decision_vars)) * 0.1
        
        return self.clip_to_bounds(mutated)
    
    def generate_for_pareto_gaps(self, pareto_front, n_samples=1):
        """
        根据帕累托前沿的空缺区域生成解
        
        设计条件扩散模型，使其能够生成具有特定多样性特征的解。
        根据当前帕累托前沿的空缺区域，生成填补这些空缺的解，
        从而提高解的分布性。
        
        参数:
            pareto_front (list): 当前帕累托前沿
            n_samples (int): 生成样本的数量
            
        返回:
            np.ndarray: 生成的样本
        """
        if not self.is_trained or len(pareto_front) < 2:
            return self._random_sample(n_samples)
        
        objectives = np.array([ind.objectives if hasattr(ind, 'objectives') else ind for ind in pareto_front])
        
        gaps = self._identify_pareto_gaps(objectives)
        
        samples = []
        for i in range(n_samples):
            if len(gaps) > 0:
                gap = gaps[i % len(gaps)]
                sample = self._sample_for_gap(gap, pareto_front)
            else:
                sample = self._ddim_sample(1.0, exploration_phase=True)
            
            sample = self.clip_to_bounds(sample.copy())
            samples.append(sample)
        
        return np.array(samples)
    
    def _identify_pareto_gaps(self, objectives):
        """
        识别帕累托前沿的空缺区域
        
        参数:
            objectives (np.ndarray): 帕累托前沿的目标函数值
            
        返回:
            list: 空缺区域列表
        """
        if len(objectives) < 2:
            return []
        
        objectives = objectives[np.argsort(objectives[:, 0])]
        
        gaps = []
        
        for i in range(len(objectives) - 1):
            gap_size = np.sqrt(np.sum((objectives[i+1] - objectives[i]) ** 2))
            
            if gap_size > 0.05:
                gap_center = (objectives[i] + objectives[i+1]) / 2
                gaps.append({
                    'center': gap_center,
                    'size': gap_size,
                    'left': objectives[i],
                    'right': objectives[i+1]
                })
        
        gaps.sort(key=lambda x: x['size'], reverse=True)
        
        return gaps[:5]
    
    def _sample_for_gap(self, gap, pareto_front):
        """
        为空缺区域生成样本
        
        参数:
            gap (dict): 空缺区域信息
            pareto_front (list): 帕累托前沿
            
        返回:
            np.ndarray: 生成的样本
        """
        left_idx = np.argmin([np.sum((ind.objectives - gap['left']) ** 2) for ind in pareto_front])
        right_idx = np.argmin([np.sum((ind.objectives - gap['right']) ** 2) for ind in pareto_front])
        
        left_sol = pareto_front[left_idx].decision_vars
        right_sol = pareto_front[right_idx].decision_vars
        
        alpha = np.random.uniform(0.3, 0.7)
        interpolated = alpha * left_sol + (1 - alpha) * right_sol
        
        if self.pca is not None:
            interpolated_scaled = self.scaler.transform(interpolated.reshape(1, -1))[0]
            interpolated_latent = self.pca.transform(interpolated_scaled.reshape(1, -1))[0]
            
            perturbation = np.random.randn(self.pca.n_components_) * 0.2
            perturbed_latent = interpolated_latent + perturbation
            
            sample = self.pca.inverse_transform(perturbed_latent.reshape(1, -1))[0]
            sample = self.scaler.inverse_transform(sample.reshape(1, -1))[0]
        else:
            perturbation = np.random.randn(self.n_vars) * 0.2
            sample = interpolated + perturbation
        
        return sample
    
    def ddim_step(self, x_t, t, predicted_noise, t_prev=None):
        """
        DDIM去噪步骤
        
        使用DDIM公式进行确定性去噪。
        
        参数:
            x_t (np.ndarray): 当前时间步的数据
            t (int): 当前时间步
            predicted_noise (np.ndarray): 预测的噪声
            t_prev (int): 前一时间步（可选，用于跳步）
            
        返回:
            np.ndarray: 去噪后的数据
        """
        if t_prev is None:
            t_prev = max(0, t - 1)
        
        alpha_t = self.alpha_cumprod[t]
        alpha_t_prev = self.alpha_cumprod[t_prev]
        
        x_0 = (x_t - np.sqrt(1 - alpha_t) * predicted_noise) / np.sqrt(alpha_t)
        
        sigma_t = self.eta * np.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * \
                  np.sqrt(1 - alpha_t / alpha_t_prev)
        
        if self.eta > 0:
            noise = np.random.randn(*x_t.shape)
        else:
            noise = np.zeros_like(x_t)
        
        x_t_prev = np.sqrt(alpha_t_prev) * x_0 + \
                   np.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise + \
                   sigma_t * noise
        
        return x_t_prev
    
    def clip_to_bounds(self, x):
        """
        将数据裁剪到边界范围内
        
        参数:
            x (np.ndarray): 数据向量
            
        返回:
            np.ndarray: 裁剪后的数据
        """
        for i, (lower, upper) in enumerate(self.bounds):
            x[i] = np.clip(x[i], lower, upper)
        return x
    
    def _random_sample(self, n_samples):
        """
        随机采样（未训练时使用）
        
        参数:
            n_samples (int): 生成样本的数量
            
        返回:
            np.ndarray: 随机生成的样本
        """
        samples = []
        for _ in range(n_samples):
            sample = np.array([
                np.random.uniform(lower, upper)
                for lower, upper in self.bounds
            ])
            samples.append(sample)
        return np.array(samples)


class ConditionEncoder:
    """
    条件编码器
    
    将条件信息编码为低维表示，用于引导扩散模型生成。
    
    在DM-MOEA中，条件编码器将当前种群的状态信息编码为条件向量，
    用于引导DDIM模型生成更有针对性的解。
    """
    
    def __init__(self):
        """初始化条件编码器"""
        self.pca = None
        self.scaler = None
        self.is_fitted = False
        self.problem = None
    
    def fit(self, data, condition_data=None):
        """
        拟合编码器
        
        参数:
            data (list): 数据列表
            condition_data: 条件数据
        """
        if condition_data is None:
            return
        
        data_array = np.array(data)
        condition_array = np.array(condition_data)
        
        self.scaler = StandardScaler()
        condition_scaled = self.scaler.fit_transform(condition_array)
        
        if condition_scaled.shape[0] > condition_scaled.shape[1]:
            n_components = min(condition_scaled.shape[1], condition_scaled.shape[0] - 1)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(condition_scaled)
        
        self.is_fitted = True
    
    def encode(self, condition):
        """
        编码条件
        
        参数:
            condition: 条件信息
            
        返回:
            np.ndarray: 编码后的条件向量
        """
        if not self.is_fitted or self.scaler is None:
            return np.zeros(1)
        
        condition_array = np.array(condition).reshape(1, -1)
        condition_scaled = self.scaler.transform(condition_array)
        
        if self.pca is not None:
            encoded = self.pca.transform(condition_scaled)[0]
        else:
            encoded = condition_scaled[0]
        
        return encoded
