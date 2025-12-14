import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import logging

from models.diffusion import DiffusionModel, UNet
from models.evolution import DiffusionEvolutionOptimizer


class ImageDataset(Dataset):
    """简单的图像数据集示例"""
    
    def __init__(self, data: torch.Tensor):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class DiffusionEvolutionTrainer:
    """扩散-进化模型训练器"""
    
    def __init__(
        self,
        model: DiffusionModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # 目录设置
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志设置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.lrs = []
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(self.device)
            
            # 前向传播
            loss = self.model(data)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for data in pbar:
                data = data.to(self.device)
                loss = self.model(data)
                total_loss += loss.item()
                
                pbar.set_postfix({'Val Loss': f'{total_loss / len(pbar):.6f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'lrs': self.lrs
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_checkpoint.pth'))
        
        # 定期保存检查点
        if epoch % 50 == 0:
            torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.lrs = checkpoint.get('lrs', [])
        
        return checkpoint['epoch']
    
    def generate_samples(self, num_samples: int = 64, image_size: int = 32, channels: int = 3) -> torch.Tensor:
        """生成样本"""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate(num_samples, image_size, channels)
        return samples
    
    def plot_training_history(self, save_path: str = None) -> None:
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 训练损失
        axes[0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 学习率
        axes[1].plot(self.lrs)
        axes[1].set_title('Learning Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('LR')
        axes[1].set_yscale('log')
        axes[1].grid(True)
        
        # 损失分布
        if len(self.train_losses) > 1:
            axes[2].hist(self.train_losses, bins=20, alpha=0.7, label='Train')
            if self.val_losses:
                axes[2].hist(self.val_losses, bins=20, alpha=0.7, label='Val')
            axes[2].set_title('Loss Distribution')
            axes[2].set_xlabel('Loss')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        generate_every: int = 50,
        evolution_interval: int = 100,
        evolution_generations: int = 20
    ) -> Dict[str, Any]:
        """完整训练流程"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录日志
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s"
            for key, value in {**train_metrics, **val_metrics}.items():
                log_msg += f" | {key}: {value:.6f}"
            self.logger.info(log_msg)
            
            # 检查是否是最佳模型
            if 'val_loss' in val_metrics and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, is_best=True)
            
            # 定期保存检查点
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
            
            # 定期生成样本
            if epoch % generate_every == 0:
                self.generate_and_save_samples(epoch)
            
            # 定期使用进化算法优化
            if epoch % evolution_interval == 0:
                self.evolution_optimize(evolution_generations)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # 绘制训练历史
        self.plot_training_history(os.path.join(self.log_dir, 'training_history.png'))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'total_time': total_time,
            'best_val_loss': best_val_loss
        }
    
    def generate_and_save_samples(self, epoch: int, num_samples: int = 64) -> None:
        """生成并保存样本"""
        try:
            samples = self.generate_samples(num_samples)
            
            # 保存样本图像
            save_path = os.path.join(self.log_dir, f'samples_epoch_{epoch}.png')
            
            # 将样本缩放到 [0, 1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            grid = make_grid(samples, nrow=8, padding=2)
            save_image(grid, save_path)
            
            self.logger.info(f"Generated samples saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate samples: {e}")
    
    def evolution_optimize(self, num_generations: int = 20) -> None:
        """使用进化算法优化模型"""
        self.logger.info(f"Starting evolution optimization for {num_generations} generations")
        
        try:
            # 创建进化优化器
            evo_optimizer = DiffusionEvolutionOptimizer(
                self.model,
                population_size=10,
                mutation_rate=0.1,
                mutation_strength=0.01,
                crossover_rate=0.7,
                device=self.device
            )
            
            # 这里需要定义一个合适的适应度函数
            # 暂时跳过实际的进化优化，只记录日志
            self.logger.info("Evolution optimization placeholder - implement fitness function")
            
        except Exception as e:
            self.logger.error(f"Evolution optimization failed: {e}")


def create_sample_data(num_samples: int = 1000, image_size: int = 32, channels: int = 3) -> torch.Tensor:
    """创建示例数据（用于测试）"""
    # 创建一些随机图像数据作为示例
    data = torch.randn(num_samples, channels, image_size, image_size)
    return data


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 模型配置
    in_channels = 3
    image_size = 32
    
    # 创建模型
    unet = UNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=128,
        time_emb_dim=512
    )
    
    diffusion_model = DiffusionModel(
        model=unet,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )
    
    # 创建示例数据
    print("Creating sample data...")
    train_data = create_sample_data(1000, image_size, in_channels)
    val_data = create_sample_data(200, image_size, in_channels)
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)
    
    # 创建训练器
    trainer = DiffusionEvolutionTrainer(
        model=diffusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=2e-4,
        weight_decay=1e-4,
        save_dir="./checkpoints",
        log_dir="./logs"
    )
    
    # 开始训练
    print("Starting training...")
    history = trainer.train(
        num_epochs=200,
        save_every=10,
        generate_every=50,
        evolution_interval=100,
        evolution_generations=20
    )
    
    print("Training completed!")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Total training time: {history['total_time']:.2f} seconds")


if __name__ == "__main__":
    main()