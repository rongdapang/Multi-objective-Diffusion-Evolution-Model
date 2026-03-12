import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

class ConditionalDiffusionModel:
    """
    条件扩散模型（DDPM），用于在潜在空间生成分子
    """
    def __init__(self, z_dim: int = 128, cond_dim: int = 3, hidden_dims: List[int] = [256, 512, 512, 256],
                 T: int = 100, noise_schedule: str = 'cosine', device: str = 'cuda'):
        """
        初始化条件扩散模型
        
        Args:
            z_dim: 潜在空间维度
            cond_dim: 条件维度
            hidden_dims: 隐藏层维度列表
            T: 时间步数
            noise_schedule: 噪声调度类型
            device: 设备类型
        """
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.hidden_dims = hidden_dims
        self.T = T
        self.noise_schedule = noise_schedule
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 初始化噪声调度
        self.betas, self.alphas, self.alpha_bars = self._setup_noise_schedule()
        
        # 构建模型
        self.model = self._build_model()
        self.model.to(self.device)
    
    def _setup_noise_schedule(self):
        """
        设置噪声调度
        """
        if self.noise_schedule == 'cosine':
            # 余弦噪声调度
            betas = []
            alpha_bars = []
            alphas = []
            
            for t in range(self.T):
                t1 = t / self.T
                t2 = (t + 1) / self.T
                alpha_bar_t1 = np.cos((t1 + 0.008) / 1.008 * np.pi / 2) ** 2
                alpha_bar_t2 = np.cos((t2 + 0.008) / 1.008 * np.pi / 2) ** 2
                beta = 1 - alpha_bar_t2 / alpha_bar_t1
                beta = np.clip(beta, 0.0001, 0.9999)
                betas.append(beta)
                alpha_bars.append(alpha_bar_t2)
                alphas.append(1 - beta)
            
            betas = np.array(betas)
            alphas = np.array(alphas)
            alpha_bars = np.array(alpha_bars)
        else:
            # 线性噪声调度
            beta_start = 0.0001
            beta_end = 0.02
            betas = np.linspace(beta_start, beta_end, self.T)
            alphas = 1 - betas
            alpha_bars = np.cumprod(alphas)
        
        return torch.tensor(betas).to(self.device), torch.tensor(alphas).to(self.device), torch.tensor(alpha_bars).to(self.device)
    
    def _build_model(self):
        """
        构建MLP模型
        """
        layers = []
        input_dim = self.z_dim + self.z_dim + self.cond_dim  # z_t + t_emb + cond
        
        # 输入层
        layers.append(nn.Linear(input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(self.hidden_dims[-1], self.z_dim))
        
        return nn.Sequential(*layers)
    
    def _positional_encoding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        正弦位置编码
        
        Args:
            t: 时间步张量
            dim: 编码维度
        """
        device = t.device
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
    
    def train(self, z_train: np.ndarray, cond_train: np.ndarray, epochs: int = 100, batch_size: int = 256):
        """
        训练模型
        
        Args:
            z_train: 训练集潜在向量
            cond_train: 训练集条件
            epochs: 训练轮数
            batch_size: 批次大小
        """
        z_train = torch.tensor(z_train, dtype=torch.float32).to(self.device)
        cond_train = torch.tensor(cond_train, dtype=torch.float32).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 随机打乱数据
            perm = torch.randperm(len(z_train))
            z_train = z_train[perm]
            cond_train = cond_train[perm]
            
            total_loss = 0
            
            for i in range(0, len(z_train), batch_size):
                batch_z = z_train[i:i+batch_size]
                batch_cond = cond_train[i:i+batch_size]
                
                # 随机采样时间步
                t = torch.randint(0, self.T, (len(batch_z),), device=self.device)
                
                # 生成噪声
                noise = torch.randn_like(batch_z)
                
                # 计算alpha_bar
                alpha_bar = self.alpha_bars[t].unsqueeze(1)
                
                # 生成带噪声的潜在向量
                z_t = torch.sqrt(alpha_bar) * batch_z + torch.sqrt(1 - alpha_bar) * noise
                
                # 计算时间步嵌入
                t_emb = self._positional_encoding(t, self.z_dim)
                
                # 拼接输入
                input = torch.cat([z_t, t_emb, batch_cond], dim=1)
                
                # 预测噪声
                predicted_noise = self.model(input)
                
                # 计算损失
                loss = criterion(predicted_noise, noise)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(z_train) / batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save('diffusion.pth')
    
    def sample(self, cond: np.ndarray, steps: int = 50) -> np.ndarray:
        """
        条件生成潜在向量
        
        Args:
            cond: 条件向量
            steps: 采样步数
        
        Returns:
            生成的潜在向量
        """
        cond = torch.tensor(cond, dtype=torch.float32).to(self.device)
        n = len(cond)
        
        # 从高斯噪声开始
        z = torch.randn(n, self.z_dim, device=self.device)
        
        # 计算采样步长
        step_size = self.T // steps
        
        for i in range(steps):
            t = self.T - i * step_size - 1
            t_tensor = torch.full((n,), t, device=self.device, dtype=torch.long)
            
            # 计算时间步嵌入
            t_emb = self._positional_encoding(t_tensor, self.z_dim)
            
            # 预测噪声
            input = torch.cat([z, t_emb, cond], dim=1)
            predicted_noise = self.model(input)
            
            # 计算alpha和beta
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            # 更新z
            if t > 0:
                noise = torch.randn_like(z)
                z = (z - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
            else:
                z = (z - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        
        return z.cpu().numpy()
    
    def save(self, path: str):
        """
        保存模型权重
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'z_dim': self.z_dim,
            'cond_dim': self.cond_dim,
            'hidden_dims': self.hidden_dims,
            'T': self.T,
            'noise_schedule': self.noise_schedule
        }, path)
    
    def load(self, path: str):
        """
        加载模型权重
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.z_dim = checkpoint['z_dim']
        self.cond_dim = checkpoint['cond_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.T = checkpoint['T']
        self.noise_schedule = checkpoint['noise_schedule']
        self.betas, self.alphas, self.alpha_bars = self._setup_noise_schedule()
