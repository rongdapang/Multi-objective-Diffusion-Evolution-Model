"""
扩散-进化模型简单示例
展示如何使用模型进行训练和生成
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import DiffusionModel, UNet
from training.trainer import DiffusionEvolutionTrainer
from utils.utils import save_samples, plot_training_curves, set_seed


def create_simple_dataset(num_samples=1000, image_size=32, channels=3):
    """创建简单的模拟数据集"""
    # 创建一些简单的图案作为训练数据
    data = []
    
    for _ in range(num_samples):
        # 创建随机图像
        img = np.random.randn(channels, image_size, image_size)
        
        # 添加一些简单的模式
        x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
        center = image_size // 2
        
        # 随机选择图案类型
        pattern_type = np.random.choice(['circle', 'square', 'noise'])
        
        if pattern_type == 'circle':
            mask = ((x - center) ** 2 + (y - center) ** 2) < (image_size // 4) ** 2
            for c in range(channels):
                img[c] = np.where(mask, np.random.randn(), img[c])
        
        elif pattern_type == 'square':
            mask = (np.abs(x - center) < image_size // 4) & (np.abs(y - center) < image_size // 4)
            for c in range(channels):
                img[c] = np.where(mask, np.random.randn(), img[c])
        
        data.append(img)
    
    data = np.array(data)
    return torch.from_numpy(data).float()


def main():
    """主函数 - 展示完整的使用流程"""
    
    print("=" * 60)
    print("扩散-进化模型简单示例")
    print("=" * 60)
    
    # 1. 设置随机种子
    set_seed(42)
    
    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 3. 创建数据集
    print("\n创建模拟数据集...")
    train_data = create_simple_dataset(500, image_size=32, channels=3)
    val_data = create_simple_dataset(100, image_size=32, channels=3)
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"验证数据形状: {val_data.shape}")
    
    # 4. 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # 5. 创建模型（使用较小的模型以加快训练）
    print("\n创建模型...")
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,  # 使用较小的模型
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(16,),
        dropout=0.1,
        time_emb_dim=256
    )
    
    model = DiffusionModel(
        model=unet,
        timesteps=500,  # 减少时间步数以加快训练
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 6. 创建训练器
    trainer = DiffusionEvolutionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,  # 使用较大的学习率
        weight_decay=1e-4,
        save_dir="./simple_example_checkpoints",
        log_dir="./simple_example_logs"
    )
    
    # 7. 训练模型（少量epoch以快速演示）
    print("\n开始训练...")
    history = trainer.train(
        num_epochs=50,  # 减少训练轮数
        save_every=10,
        generate_every=25,
        evolution_interval=50,  # 在这个简单的例子中不使用进化优化
        evolution_generations=10
    )
    
    # 8. 生成样本
    print("\n生成样本...")
    samples = trainer.generate_samples(num_samples=16, image_size=32, channels=3)
    
    # 9. 保存结果
    os.makedirs("./simple_example_results", exist_ok=True)
    
    # 保存生成的样本
    save_samples(
        samples=samples,
        save_path="./simple_example_results/generated_samples.png",
        nrow=4,
        title="Generated Samples"
    )
    
    # 绘制训练曲线
    plot_training_curves(
        train_losses=history['train_losses'],
        val_losses=history.get('val_losses', []),
        save_path="./simple_example_results/training_curves.png",
        title="Training Curves"
    )
    
    # 10. 保存模型
    torch.save(model.state_dict(), "./simple_example_results/diffusion_model.pth")
    
    print("\n示例完成！")
    print("生成的文件:")
    print("  - ./simple_example_results/generated_samples.png")
    print("  - ./simple_example_results/training_curves.png")
    print("  - ./simple_example_results/diffusion_model.pth")
    
    # 显示一些统计信息
    print(f"\n训练统计:")
    print(f"  - 总训练时间: {history['total_time']:.2f} 秒")
    print(f"  - 最佳验证损失: {history['best_val_loss']:.6f}")
    print(f"  - 最终训练损失: {history['train_losses'][-1]:.6f}")


# 一个更简单的快速演示版本
def quick_demo():
    """快速演示 - 用于快速验证代码"""
    
    print("\n快速演示模式...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建非常小的模型
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_emb_dim=128
    )
    
    model = DiffusionModel(
        model=unet,
        timesteps=100,  # 非常少的时间步
        device=device
    )
    
    # 创建虚拟数据
    dummy_data = torch.randn(20, 3, 32, 32).to(device)
    train_loader = DataLoader(dummy_data, batch_size=4)
    
    # 创建训练器
    trainer = DiffusionEvolutionTrainer(
        model=model,
        train_loader=train_loader,
        device=device,
        lr=1e-3
    )
    
    # 快速训练几轮
    print("快速训练5轮...")
    history = trainer.train(num_epochs=5)
    
    # 生成样本
    samples = trainer.generate_samples(num_samples=4)
    
    print(f"快速演示完成！")
    print(f"最终损失: {history['train_losses'][-1]:.4f}")
    print(f"生成样本形状: {samples.shape}")


if __name__ == "__main__":
    import sys
    
    # 如果提供了 --quick 参数，运行快速演示
    if '--quick' in sys.argv:
        quick_demo()
    else:
        main()