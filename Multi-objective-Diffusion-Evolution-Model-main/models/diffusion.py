import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块，用于UNet架构"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # 确保time_emb的维度与h匹配
        time_emb = self.time_mlp(time_emb)
        # 调整time_emb的形状以匹配h
        while len(time_emb.shape) < len(h.shape):
            time_emb = time_emb.unsqueeze(-1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    
    def __init__(self, channels: int, size: int):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).swapaxes(1, 2)  # (B, H*W, C)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(B, C, H, W)


class UNet(nn.Module):
    """简化的UNet架构"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 编码器
        self.enc1 = ResidualBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        # 中间层
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # 解码器
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # 输出层
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
        # 下采样和上采样
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入图像
            timesteps: (B,) 时间步
        Returns:
            (B, C, H, W) 预测的噪声
        """
        # 时间嵌入
        time_emb = self.time_embed(timesteps)
        
        # 编码
        h1 = self.enc1(x, time_emb)
        h2 = self.enc2(self.downsample(h1), time_emb)
        h3 = self.enc3(self.downsample(h2), time_emb)
        
        # 中间层
        h = self.mid(self.downsample(h3), time_emb)
        
        # 解码
        h = self.upsample(h)
        h = torch.cat([h, h3], dim=1)
        h = self.dec3(h, time_emb)
        
        h = self.upsample(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, time_emb)
        
        h = self.upsample(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, time_emb)
        
        return self.conv_out(h)


class DiffusionModel(nn.Module):
    """扩散模型"""
    
    def __init__(
        self,
        model: UNet,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: torch.device = None
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 定义beta调度
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # 预计算值
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)
        
        # 反向过程参数
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向过程 - 添加噪声"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alpha_hat_t * x0 + sqrt_one_minus_alpha_hat_t * noise
    
    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        
        # MSE损失
        loss = F.mse_loss(noise, predicted_noise, reduction='mean')
        return loss
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """反向采样单步"""
        betas_t = self.beta[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alpha_hat[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alpha[t]).reshape(-1, 1, 1, 1)
        
        # 方程11的系数
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape: torch.Size) -> torch.Tensor:
        """完整的反向采样过程"""
        device = self.device
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        
        return x
    
    def generate(self, num_samples: int, image_size: int = 32, channels: int = 3) -> torch.Tensor:
        """生成样本"""
        shape = (num_samples, channels, image_size, image_size)
        return self.p_sample_loop(shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播用于训练"""
        B = x.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x.device).long()
        return self.p_losses(x, t)