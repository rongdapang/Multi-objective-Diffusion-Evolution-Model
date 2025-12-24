"""
扩散模型实现

基于DDPM (Denoising Diffusion Probabilistic Models) 和 DDIM (Denoising Diffusion Implicit Models)
的简化实现，专门用于多目标优化中的解生成。
"""

import numpy as np
import warnings
from typing import Optional, Tuple


class NoiseScheduler:
    """噪声调度器 - 管理扩散过程中的噪声添加"""
    
    def __init__(self, timesteps: int = 1000, schedule_type: str = 'linear',
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        初始化噪声调度器
        
        参数:
            timesteps: 扩散步数
            schedule_type: 调度类型 ('linear' 或 'cosine')
            beta_start: 起始噪声水平
            beta_end: 结束噪声水平
        """
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # 创建噪声调度
        self._create_schedule()
        
    def _create_schedule(self):
        """创建噪声调度"""
        if self.schedule_type == 'linear':
            # 线性调度
            self.betas = np.linspace(self.beta_start, self.beta_end, self.timesteps)
        elif self.schedule_type == 'cosine':
            # 余弦调度
            s = 0.008
            steps = np.linspace(0, self.timesteps, self.timesteps + 1)
            ft = np.cos((steps / self.timesteps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = ft / ft[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"未知的调度类型: {self.schedule_type}")
            
        # 预计算参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于反向过程的参数
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def add_noise(self, x0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向过程 - 向数据添加噪声
        
        参数:
            x0: 原始数据 (batch_size, dim)
            t: 时间步 (batch_size,)
            noise: 噪声 (batch_size, dim)，如果为None则随机生成
            
        返回:
            加噪后的数据
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)
            
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
    def sample_timesteps(self, batch_size: int) -> np.ndarray:
        """随机采样时间步"""
        return np.random.randint(0, self.timesteps, size=(batch_size,))


class SimpleMLP:
    """简单的多层感知机，用于噪声预测"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, time_embed_dim: int = 32):
        """
        初始化MLP
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            time_embed_dim: 时间嵌入维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # 初始化权重（简化版本）
        self.time_embed_weight = np.random.randn(time_embed_dim, time_embed_dim) * 0.1
        self.time_embed_bias = np.zeros(time_embed_dim)
        
        self.layer1_weight = np.random.randn(input_dim + time_embed_dim, hidden_dim) * 0.1
        self.layer1_bias = np.zeros(hidden_dim)
        
        self.layer2_weight = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.layer2_bias = np.zeros(hidden_dim)
        
        self.output_weight = np.random.randn(hidden_dim, input_dim) * 0.1
        self.output_bias = np.zeros(input_dim)
        
    def time_embedding(self, timesteps: np.ndarray, max_period: int = 10000) -> np.ndarray:
        """正弦时间嵌入"""
        half_dim = self.time_embed_dim // 2
        emb = np.log(max_period) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
        return emb
        
    def forward(self, x: np.ndarray, timesteps: np.ndarray, 
                conditions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入数据 (batch_size, input_dim)
            timesteps: 时间步 (batch_size,)
            conditions: 条件信息 (batch_size, condition_dim)
            
        返回:
            预测的噪声 (batch_size, input_dim)
        """
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb @ self.time_embed_weight + self.time_embed_bias
        time_emb = np.maximum(time_emb, 0)  # ReLU激活
        
        # 拼接输入
        x_cond = np.concatenate([x, time_emb], axis=1)
        
        # 第一层
        h = x_cond @ self.layer1_weight + self.layer1_bias
        h = np.maximum(h, 0)  # ReLU激活
        
        # 第二层
        h = h @ self.layer2_weight + self.layer2_bias
        h = np.maximum(h, 0)  # ReLU激活
        
        # 输出层
        noise_pred = h @ self.output_weight + self.output_bias
        
        # 添加条件信息
        if conditions is not None:
            condition_weight = np.mean(conditions, axis=1, keepdims=True)
            noise_pred = noise_pred * (1 + 0.1 * condition_weight)
            
        return noise_pred
        
    def train_step(self, x0: np.ndarray, timesteps: np.ndarray], 
                   noise: np.ndarray, learning_rate: float = 0.001) -> float:
        """
        训练一步
        
        参数:
            x0: 原始数据
            timesteps: 时间步
            noise: 真实噪声
            learning_rate: 学习率
            
        返回:
            损失值
        """
        # 前向传播
        noise_pred = self.forward(x0, timesteps)
        
        # 计算损失
        loss = np.mean((noise - noise_pred) ** 2)
        
        # 简化梯度更新（在实际实现中会使用反向传播）
        grad_scale = learning_rate * (noise_pred - noise) / len(x0)
        
        # 更新权重（简化版本）
        self.output_weight -= grad_scale[:, None] * np.ones_like(self.output_weight)
        
        return loss


class DiffusionModel:
    """扩散模型 - 用于生成高质量解"""
    
    def __init__(self, input_dim: int, noise_scheduler: NoiseScheduler, 
                 model_type: str = 'DDPM', hidden_dim: int = 64):
        """
        初始化扩散模型
        
        参数:
            input_dim: 输入维度（决策变量数量）
            noise_scheduler: 噪声调度器
            model_type: 模型类型 ('DDPM' 或 'DDIM')
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.noise_scheduler = noise_scheduler
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        # 初始化噪声预测网络
        self.network = SimpleMLP(input_dim, hidden_dim)
        
        # 训练历史
        self.training_loss = []
        
    def train_step(self, x0: np.ndarray, conditions: Optional[np.ndarray] = None) -> float:
        """
        训练一步
        
        参数:
            x0: 原始数据 (batch_size, input_dim)
            conditions: 条件信息 (batch_size, condition_dim)
            
        返回:
            损失值
        """
        batch_size = len(x0)
        
        # 采样时间步
        timesteps = self.noise_scheduler.sample_timesteps(batch_size)
        
        # 生成噪声
        noise = np.random.randn(*x0.shape)
        
        # 前向过程
        x_noisy = self.noise_scheduler.add_noise(x0, timesteps, noise)
        
        # 预测噪声
        noise_pred = self.network.forward(x_noisy, timesteps, conditions)
        
        # 计算损失
        loss = np.mean((noise - noise_pred) ** 2)
        
        # 简化训练更新
        self.network.train_step(x_noisy, timesteps, noise)
        
        self.training_loss.append(loss)
        return loss
        
    def sample(self, n_samples: int, input_dim: int) -> np.ndarray:
        """
        采样生成新解
        
        参数:
            n_samples: 采样数量
            input_dim: 输入维度
            
        返回:
            生成的样本 (n_samples, input_dim)
        """
        if self.model_type == 'DDPM':
            return self._sample_ddpm(n_samples, input_dim)
        elif self.model_type == 'DDIM':
            return self._sample_ddim(n_samples, input_dim)
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")
            
    def _sample_ddpm(self, n_samples: int, input_dim: int) -> np.ndarray:
        """DDPM采样"""
        # 从纯噪声开始
        x = np.random.randn(n_samples, input_dim)
        
        # 逐步去噪
        for t in reversed(range(self.noise_scheduler.timesteps)):
            timesteps = np.full(n_samples, t)
            
            # 预测噪声
            noise_pred = self.network.forward(x, timesteps)
            
            # 计算参数
            alpha_t = self.noise_scheduler.alphas[t]
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
            beta_t = self.noise_scheduler.betas[t]
            
            if t > 0:
                alpha_cumprod_t_prev = self.noise_scheduler.alphas_cumprod[t-1]
            else:
                alpha_cumprod_t_prev = 1.0
                
            # 计算均值
            sqrt_recip_alpha_t = np.sqrt(1.0 / alpha_t)
            sqrt_one_minus_alpha_cumprod_t = np.sqrt(1.0 - alpha_cumprod_t)
            
            mean = sqrt_recip_alpha_t * (x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)
            
            if t > 0:
                # 添加采样噪声
                posterior_variance_t = beta_t * (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)
                noise = np.random.randn(*x.shape)
                x = mean + np.sqrt(posterior_variance_t) * noise
            else:
                x = mean
                
        return x
        
    def _sample_ddim(self, n_samples: int, input_dim: int) -> np.ndarray:
        """DDIM采样（确定性采样）"""
        # DDIM采样通常使用更少的步骤
        ddim_steps = min(50, self.noise_scheduler.timesteps)
        step_size = self.noise_scheduler.timesteps // ddim_steps
        timesteps = np.arange(0, self.noise_scheduler.timesteps, step_size)
        
        # 从纯噪声开始
        x = np.random.randn(n_samples, input_dim)
        
        # 逐步去噪
        for i in reversed(range(len(timesteps))):
            t = timesteps[i]
            timesteps_batch = np.full(n_samples, t)
            
            # 预测噪声
            noise_pred = self.network.forward(x, timesteps_batch)
            
            # DDIM更新（确定性）
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
            sqrt_alpha_cumprod_t = np.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = np.sqrt(1.0 - alpha_cumprod_t)
            
            # 预测原始数据
            x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
            
            if i > 0:
                t_prev = timesteps[i-1]
                alpha_cumprod_t_prev = self.noise_scheduler.alphas_cumprod[t_prev]
                sqrt_alpha_cumprod_t_prev = np.sqrt(alpha_cumprod_t_prev)
                sqrt_one_minus_alpha_cumprod_t_prev = np.sqrt(1.0 - alpha_cumprod_t_prev)
                
                # DDIM更新
                x = sqrt_alpha_cumprod_t_prev * x0_pred + sqrt_one_minus_alpha_cumprod_t_prev * noise_pred
            else:
                x = x0_pred
                
        return x