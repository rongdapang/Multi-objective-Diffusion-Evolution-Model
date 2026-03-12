"""
条件扩散模型模块 - 用于生成潜在空间向量
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from tqdm import tqdm


class ConditionalDiffusionModel:
    """
    条件扩散模型，用于生成满足目标性质的潜在向量
    """
    
    def __init__(self, z_dim: int = 128, cond_dim: int = 3, 
                 hidden_dims: list = [256, 512, 512, 256],
                 num_diffusion_steps: int = 50,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 device: str = 'cuda'):
        """
        初始化条件扩散模型
        
        Args:
            z_dim: 潜在空间维度
            cond_dim: 条件维度（目标性质数量）
            hidden_dims: 隐藏层维度列表
            num_diffusion_steps: 扩散步数
            beta_start: 初始噪声水平
            beta_end: 最终噪声水平
            device: 计算设备
        """
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化噪声网络
        self.noise_net = ConditionalNoiseNet(
            z_dim=z_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            num_steps=num_diffusion_steps
        ).to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.noise_net.parameters(), lr=1e-3)
        
        # 设置扩散参数
        self.betas = torch.linspace(beta_start, beta_end, num_diffusion_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.is_trained = False
        
    def train(self, z_train: np.ndarray, cond_train: np.ndarray, 
              epochs: int = 100, batch_size: int = 64) -> list:
        """
        训练扩散模型
        
        Args:
            z_train: 训练用的潜在向量 (shape: [n, z_dim])
            cond_train: 训练用的条件向量 (shape: [n, cond_dim])
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练损失历史
        """
        # 转换数据为张量
        z_tensor = torch.FloatTensor(z_train).to(self.device)
        cond_tensor = torch.FloatTensor(cond_train).to(self.device)
        
        # 归一化条件
        self.cond_mean = cond_tensor.mean(dim=0)
        self.cond_std = cond_tensor.std(dim=0) + 1e-8
        cond_normalized = (cond_tensor - self.cond_mean) / self.cond_std
        
        # 归一化潜在向量
        self.z_mean = z_tensor.mean(dim=0)
        self.z_std = z_tensor.std(dim=0) + 1e-8
        z_normalized = (z_tensor - self.z_mean) / self.z_std
        
        dataset = torch.utils.data.TensorDataset(z_normalized, cond_normalized)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        losses = []
        
        self.noise_net.train()
        for epoch in tqdm(range(epochs), desc="Training diffusion model"):
            epoch_losses = []
            
            for z_batch, cond_batch in dataloader:
                # 随机选择扩散步
                t = torch.randint(0, self.num_diffusion_steps, (z_batch.shape[0],)).to(self.device)
                
                # 前向扩散
                noise = torch.randn_like(z_batch)
                z_t = self._q_sample(z_batch, t, noise)
                
                # 预测噪声
                noise_pred = self.noise_net(z_t, cond_batch, t)
                
                # 计算损失
                loss = F.mse_loss(noise_pred, noise)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.noise_net.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        return losses
    
    def _q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        前向扩散过程
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample(self, conditions: np.ndarray, num_samples: Optional[int] = None) -> np.ndarray:
        """
        从条件生成潜在向量
        
        Args:
            conditions: 目标条件 (shape: [n, cond_dim] 或 [cond_dim])
            num_samples: 样本数量（如果conditions是一维的）
            
        Returns:
            生成的潜在向量 (shape: [n, z_dim])
        """
        if not self.is_trained:
            print("警告: 扩散模型未训练，使用随机采样")
            if conditions.ndim == 1:
                return np.random.randn(num_samples if num_samples else 1, self.z_dim)
            else:
                return np.random.randn(conditions.shape[0], self.z_dim)
        
        # 处理输入
        if conditions.ndim == 1:
            if num_samples is None:
                num_samples = 1
            conditions = np.tile(conditions, (num_samples, 1))
        else:
            num_samples = conditions.shape[0]
        
        # 转换条件
        cond_tensor = torch.FloatTensor(conditions).to(self.device)
        cond_normalized = (cond_tensor - self.cond_mean.to(self.device)) / self.cond_std.to(self.device)
        
        # 从噪声开始
        z = torch.randn(num_samples, self.z_dim).to(self.device)
        
        self.noise_net.eval()
        with torch.no_grad():
            # 反向扩散
            for t in reversed(range(self.num_diffusion_steps)):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                
                # 预测噪声
                noise_pred = self.noise_net(z, cond_normalized, t_batch)
                
                # 计算均值
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
                
                mean = coef1 * (z - coef2 * noise_pred)
                
                # 添加噪声（除了最后一步）
                if t > 0:
                    noise = torch.randn_like(z)
                    variance = beta_t
                    z = mean + torch.sqrt(variance) * noise
                else:
                    z = mean
        
        # 反归一化
        z = z * self.z_std.to(self.device) + self.z_mean.to(self.device)
        
        return z.cpu().numpy()


class ConditionalNoiseNet(nn.Module):
    """
    条件噪声预测网络
    """
    
    def __init__(self, z_dim: int, cond_dim: int, hidden_dims: list, num_steps: int):
        """
        初始化噪声网络
        
        Args:
            z_dim: 潜在空间维度
            cond_dim: 条件维度
            hidden_dims: 隐藏层维度列表
            num_steps: 扩散步数
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        
        # 时间步嵌入
        self.time_embed = nn.Embedding(num_steps, 128)
        
        # 条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 构建网络层
        layers = []
        input_dim = z_dim + 128 + 128  # z + time_embed + cond_embed
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, z_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 带噪声的潜在向量 (shape: [batch, z_dim])
            cond: 条件向量 (shape: [batch, cond_dim])
            t: 时间步 (shape: [batch])
            
        Returns:
            预测的噪声 (shape: [batch, z_dim])
        """
        # 时间嵌入
        t_embed = self.time_embed(t)
        
        # 条件投影
        cond_embed = self.cond_proj(cond)
        
        # 拼接特征
        x = torch.cat([z, t_embed, cond_embed], dim=1)
        
        # 通过网络
        noise_pred = self.network(x)
        
        return noise_pred
