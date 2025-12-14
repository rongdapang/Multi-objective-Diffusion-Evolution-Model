#!/usr/bin/env python3
"""
扩散-进化模型演示脚本
展示项目的核心功能和用法
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.diffusion import DiffusionModel, UNet
from models.evolution import GeneticAlgorithm, TournamentSelection
from training.trainer import DiffusionEvolutionTrainer
from configs.config import ConfigManager
from utils.utils import set_seed, save_samples, plot_training_curves


def print_banner():
    """打印项目横幅"""
    banner = """
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║           扩散-进化模型 (Diffusion-Evolution Model)                    ║
║                                                                        ║
║        结合扩散模型和进化算法的深度学习框架                           ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_diffusion_model():
    """演示扩散模型"""
    print("\n" + "=" * 80)
    print("演示 1: 扩散模型 (Diffusion Model)")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(42)
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")
    
    # 创建模型
    print("\n🔧 创建模型...")
    unet = UNet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256)
    model = DiffusionModel(model=unet, timesteps=200, device=device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数数量: {total_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 创建随机数据
    print("\n📊 创建模拟数据...")
    x = torch.randn(4, 3, 32, 32).to(device)
    print(f"   输入形状: {x.shape}")
    
    # 测试前向传播
    print("\n🔄 测试前向传播...")
    loss = model(x)
    print(f"   损失值: {loss.item():.4f}")
    
    # 生成样本
    print("\n🎨 生成样本...")
    with torch.no_grad():
        samples = model.generate(4, 32, 3)
    print(f"   生成样本形状: {samples.shape}")
    
    # 保存样本
    os.makedirs("./demo_results", exist_ok=True)
    save_samples(samples, "./demo_results/diffusion_samples.png", nrow=2)
    print(f"   样本已保存到: ./demo_results/diffusion_samples.png")
    
    return model


def demo_evolution_algorithm():
    """演示进化算法"""
    print("\n" + "=" * 80)
    print("演示 2: 进化算法 (Evolutionary Algorithm)")
    print("=" * 80)
    
    # 定义适应度函数 (寻找目标值)
    def fitness_function(genes):
        target = torch.ones_like(genes) * 0.5
        fitness = -torch.sum((genes - target) ** 2).item()
        return fitness
    
    print("\n🎯 目标: 找到接近 [0.5, 0.5, ..., 0.5] 的基因")
    print("🧬 基因长度: 10")
    print("👥 种群大小: 30")
    print("🔄 进化代数: 30")
    
    # 创建遗传算法
    ga = GeneticAlgorithm(
        population_size=30,
        gene_length=10,
        fitness_function=fitness_function,
        selection_strategy=TournamentSelection(tournament_size=3),
        mutation_rate=0.1,
        mutation_strength=0.1,
        crossover_rate=0.8,
        elitism=2,
        maximize=True
    )
    
    print("\n🚀 开始进化...")
    best_individual = ga.run(num_generations=30, target_fitness=-0.01)
    
    print(f"\n🏆 进化结果:")
    print(f"   最佳适应度: {best_individual.fitness:.6f}")
    print(f"   最佳基因: {best_individual.genes.numpy()}")
    print(f"   目标基因: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]")
    
    # 可视化进化过程
    stats = ga.get_statistics()
    if stats['best_fitness'] is not None:
        plt.figure(figsize=(12, 4))
        
        # 适应度曲线
        plt.subplot(1, 2, 1)
        generations = range(len(ga.population.history['best_fitness']))
        plt.plot(generations, ga.population.history['best_fitness'], 'g-', label='Best Fitness', linewidth=2)
        plt.plot(generations, ga.population.history['average_fitness'], 'b--', label='Average Fitness', linewidth=2)
        plt.title('Fitness Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 多样性曲线
        plt.subplot(1, 2, 2)
        plt.plot(generations, ga.population.history['diversity'], 'purple', linewidth=2)
        plt.title('Population Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./demo_results/evolution_stats.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   统计图表已保存到: ./demo_results/evolution_stats.png")
    
    return ga


def demo_training():
    """演示训练过程"""
    print("\n" + "=" * 80)
    print("演示 3: 训练过程 (Training Process)")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(42)
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")
    
    # 创建虚拟数据集
    print("\n📊 创建虚拟数据集...")
    def create_pattern_data(num_samples, image_size=32):
        """创建带有简单图案的数据"""
        data = []
        for _ in range(num_samples):
            # 创建随机图像
            img = np.random.randn(3, image_size, image_size) * 0.5
            
            # 添加简单图案
            x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
            center = image_size // 2
            
            # 随机选择图案
            pattern = np.random.choice(['circle', 'square'])
            
            if pattern == 'circle':
                mask = ((x - center) ** 2 + (y - center) ** 2) < (image_size // 4) ** 2
                for c in range(3):
                    img[c] = np.where(mask, img[c] + 1.0, img[c])
            else:
                mask = (np.abs(x - center) < image_size // 4) & (np.abs(y - center) < image_size // 4)
                for c in range(3):
                    img[c] = np.where(mask, img[c] + 1.0, img[c])
            
            data.append(img)
        
        return torch.from_numpy(np.array(data)).float()
    
    train_data = create_pattern_data(100)
    val_data = create_pattern_data(40)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=False)
    
    print(f"   训练数据: {len(train_data)} 个样本")
    print(f"   验证数据: {len(val_data)} 个样本")
    print(f"   数据形状: {train_data.shape}")
    
    # 创建模型
    print("\n🔧 创建模型...")
    unet = UNet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256)
    model = DiffusionModel(model=unet, timesteps=100, device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")
    
    # 创建训练器
    print("\n🏋️ 创建训练器...")
    trainer = DiffusionEvolutionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        save_dir="./demo_checkpoints",
        log_dir="./demo_logs"
    )
    
    # 开始训练
    print("\n🚀 开始训练 (20轮)...")
    print("   这可能需要几分钟时间...")
    
    history = trainer.train(
        num_epochs=20,
        save_every=10,
        generate_every=20,
        evolution_interval=50,  # 在这个演示中不使用进化优化
        evolution_generations=10
    )
    
    print(f"\n🏁 训练完成！")
    print(f"   总训练时间: {history['total_time']:.2f} 秒")
    print(f"   最终训练损失: {history['train_losses'][-1]:.4f}")
    print(f"   最佳验证损失: {history['best_val_loss']:.4f}")
    
    # 生成样本
    print("\n🎨 生成最终样本...")
    samples = trainer.generate_samples(num_samples=16, image_size=32, channels=3)
    save_samples(samples, "./demo_results/final_samples.png", nrow=4)
    print(f"   最终样本已保存到: ./demo_results/final_samples.png")
    
    # 保存训练曲线
    if history['train_losses']:
        plot_training_curves(
            train_losses=history['train_losses'],
            val_losses=history.get('val_losses', []),
            save_path="./demo_results/training_curves.png",
            title="Training Curves"
        )
        print(f"   训练曲线已保存到: ./demo_results/training_curves.png")
    
    return trainer


def main():
    """主函数"""
    print_banner()
    
    # 创建结果目录
    os.makedirs("./demo_results", exist_ok=True)
    
    print("\n" + "=" * 80)
    print("扩散-进化模型演示")
    print("=" * 80)
    print("\n这个演示将展示项目的三个核心功能：")
    print("  1. 扩散模型的前向传播和样本生成")
    print("  2. 进化算法的优化过程")
    print("  3. 完整的训练流程")
    print("\n演示大约需要5-10分钟完成...")
    
    # 演示1: 扩散模型
    model = demo_diffusion_model()
    
    # 演示2: 进化算法
    ga = demo_evolution_algorithm()
    
    # 演示3: 训练过程
    trainer = demo_training()
    
    # 总结
    print("\n" + "=" * 80)
    print("演示总结")
    print("=" * 80)
    print("\n✅ 所有演示已完成！")
    print("\n生成的文件：")
    print("  📊 ./demo_results/diffusion_samples.png - 扩散模型生成的样本")
    print("  📈 ./demo_results/evolution_stats.png - 进化算法统计图表")
    print("  🎨 ./demo_results/final_samples.png - 训练后生成的样本")
    print("  📊 ./demo_results/training_curves.png - 训练曲线")
    print("\n项目特色：")
    print("  ✨ 完整的扩散模型实现")
    print("  🧬 多种进化算法策略")
    print("  🚀 完整的训练系统")
    print("  ⚙️  灵活的配置管理")
    print("  📊 丰富的可视化工具")
    print("\n下一步建议：")
    print("  1. 查看 README.md 了解项目详情")
    print("  2. 阅读 GETTING_STARTED.md 学习使用方法")
    print("  3. 运行 test_all.py 进行完整测试")
    print("  4. 尝试使用自己的数据集进行训练")
    print("=" * 80)


if __name__ == "__main__":
    main()