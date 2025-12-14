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
    """扩散模型使用的UNet架构"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        time_emb_dim: int = 512
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_resolutions = len(channel_mults)
        self.num_res_blocks = num_res_blocks
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样层
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        in_ch = base_channels
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
                
                # 添加注意力层
                if 32 // (2 ** level) in attention_resolutions:
                    self.down_attentions.append(AttentionBlock(out_ch, 32 // (2 ** level)))
                else:
                    self.down_attentions.append(nn.Identity())
            
            # 下采样（最后一层不下采样）
            if level < self.num_resolutions - 1:
                self.down_blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
                self.down_attentions.append(nn.Identity())
        
        # 中间层
        mid_channels = base_channels * channel_mults[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_channels, 32 // (2 ** (self.num_resolutions - 1)))
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, dropout)
        
        # 上采样层
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                # 确保通道数不会太大
                actual_out_ch = min(out_ch, 512)
                self.up_blocks.append(ResidualBlock(in_ch + out_ch, actual_out_ch, time_emb_dim, dropout))
                in_ch = actual_out_ch
                
                # 添加注意力层
                if 32 // (2 ** (self.num_resolutions - 1 - level)) in attention_resolutions:
                    self.up_attentions.append(AttentionBlock(actual_out_ch, 32 // (2 ** (self.num_resolutions - 1 - level))))
                else:
                    self.up_attentions.append(nn.Identity())
            
            # 上采样（最后一层不上采样）
            if level < self.num_resolutions - 1:
                self.up_blocks.append(nn.ConvTranspose2d(actual_out_ch, actual_out_ch, 4, stride=2, padding=1))
                self.up_attentions.append(nn.Identity())
        
        # 输出层
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, 3, padding=1)
        )
    
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
        
        # 初始卷积
        h = self.conv_in(x)
        
        # 存储跳跃连接
        hs = [h]
        
        # 下采样
        for i, (block, attn) in enumerate(zip(self.down_blocks, self.down_attentions)):
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
                h = attn(h)
            else:  # 下采样层
                h = block(h)
            hs.append(h)
        
        # 中间层
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # 上采样
        for i, (block, attn) in enumerate(zip(self.up_blocks, self.up_attentions)):
            if isinstance(block, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, time_emb)
                h = attn(h)
            else:  # 上采样层
                h = block(h)
        
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
        betas_t = self.beta[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alpha_hat[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alpha[t])
        
        # 方程11的系数
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
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