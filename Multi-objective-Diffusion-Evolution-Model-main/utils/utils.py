import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from typing import List, Tuple, Optional, Union
import os
import time
from datetime import datetime
import logging


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('diffusion_evolution')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的handlers
    logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    path: str,
    **kwargs
) -> None:
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        **kwargs
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> dict:
    """加载模型检查点"""
    return torch.load(path, map_location=map_location)


def create_sample_grid(
    samples: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    scale_each: bool = False,
    pad_value: float = 0.0
) -> torch.Tensor:
    """创建样本网格图像"""
    if normalize:
        # 将样本从 [-1, 1] 缩放到 [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
    
    grid = vutils.make_grid(
        samples,
        nrow=nrow,
        padding=padding,
        normalize=False,  # 已经手动归一化了
        scale_each=scale_each,
        pad_value=pad_value
    )
    
    return grid


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    title: Optional[str] = None
) -> None:
    """保存生成的样本"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    grid = create_sample_grid(samples, nrow=nrow)
    
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title(title or "Generated Samples")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Curves"
) -> None:
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.8)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.8)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失分布
    plt.subplot(1, 2, 2)
    plt.hist(train_losses, bins=20, alpha=0.7, label='Train', density=True)
    if val_losses:
        plt.hist(val_losses, bins=20, alpha=0.7, label='Val', density=True)
    
    plt.title('Loss Distribution')
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_evolution_statistics(
    best_fitness: List[float],
    avg_fitness: List[float],
    diversity: List[float],
    save_path: Optional[str] = None
) -> None:
    """绘制进化算法统计信息"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    generations = range(1, len(best_fitness) + 1)
    
    # 适应度曲线
    axes[0].plot(generations, best_fitness, 'g-', label='Best Fitness', linewidth=2)
    axes[0].plot(generations, avg_fitness, 'b--', label='Average Fitness', linewidth=2)
    axes[0].set_title('Fitness Evolution')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 多样性
    axes[1].plot(generations, diversity, 'purple', linewidth=2)
    axes[1].set_title('Population Diversity')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Diversity')
    axes[1].grid(True, alpha=0.3)
    
    # 适应度分布
    axes[2].hist(best_fitness, bins=15, alpha=0.7, label='Best', density=True)
    axes[2].hist(avg_fitness, bins=15, alpha=0.7, label='Average', density=True)
    axes[2].set_title('Fitness Distribution')
    axes[2].set_xlabel('Fitness')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_model_size(model: nn.Module) -> dict:
    """计算模型大小信息"""
    param_size = 0
    param_count = 0
    buffer_size = 0
    buffer_count = 0
    
    for param in model.parameters():
        param_count += 1
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_count += 1
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'param_count': param_count,
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_count': buffer_count,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': size_all_mb
    }


def print_model_info(model: nn.Module, name: str = "Model") -> None:
    """打印模型信息"""
    print("=" * 60)
    print(f"{name.upper()} INFORMATION")
    print("=" * 60)
    
    # 计算模型大小
    model_info = calculate_model_size(model)
    
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {model_info['total_size_mb']:.2f} MB")
    print(f"Parameters size: {model_info['param_size_mb']:.2f} MB")
    print(f"Buffers size: {model_info['buffer_size_mb']:.2f} MB")
    
    # 打印模型结构（简化版）
    print("\nModel Architecture:")
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            if params > 0:
                print(f"  {name}: {params:,} parameters")
    
    print("=" * 60)


def get_memory_usage() -> dict:
    """获取内存使用情况"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # GPU 内存
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        
        return {
            'gpu_allocated_mb': gpu_memory_allocated,
            'gpu_cached_mb': gpu_memory_cached
        }
    else:
        return {'gpu_allocated_mb': 0, 'gpu_cached_mb': 0}


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def create_progress_bar(current: int, total: int, prefix: str = "", length: int = 50) -> str:
    """创建进度条"""
    percent = float(current) / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    return f"{prefix} |{bar}| {percent * 100:.1f}% ({current}/{total})"


# 数据增强工具
class ImageAugmentation:
    """图像数据增强工具"""
    
    @staticmethod
    def random_flip(images: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """随机水平翻转"""
        if torch.rand(1) < p:
            return torch.flip(images, [3])  # 翻转宽度维度
        return images
    
    @staticmethod
    def random_rotation(images: torch.Tensor, max_angle: float = 15.0) -> torch.Tensor:
        """随机旋转（简化版）"""
        # 这里应该使用torchvision的transforms，为了简化暂时跳过
        return images
    
    @staticmethod
    def color_jitter(images: torch.Tensor, brightness: float = 0.1, contrast: float = 0.1) -> torch.Tensor:
        """颜色抖动"""
        # 亮度调整
        if brightness > 0:
            brightness_factor = 1 + torch.rand(1).item() * brightness * 2 - brightness
            images = images * brightness_factor
        
        # 对比度调整
        if contrast > 0:
            contrast_factor = 1 + torch.rand(1).item() * contrast * 2 - contrast
            images = images * contrast_factor
        
        return torch.clamp(images, -1, 1)  # 假设输入是[-1, 1]


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将tensor转换为numpy数组"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    return tensor.numpy()


def numpy_to_tensor(array: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """将numpy数组转换为tensor"""
    tensor = torch.from_numpy(array)
    
    if device:
        tensor = tensor.to(device)
    
    return tensor.float()