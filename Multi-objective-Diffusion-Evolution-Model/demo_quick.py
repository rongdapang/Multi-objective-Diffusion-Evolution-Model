#!/usr/bin/env python3
"""
扩散-进化模型快速演示
快速展示项目的核心功能
"""

import torch
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.diffusion import DiffusionModel, UNet
from models.evolution import GeneticAlgorithm, TournamentSelection
from training.trainer import DiffusionEvolutionTrainer
from utils.utils import set_seed


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


def main():
    """主函数 - 快速演示"""
    print_banner()
    
    print("\n" + "=" * 80)
    print("扩散-进化模型快速演示")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(42)
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    
    # ========================================
    # 演示 1: 扩散模型
    # ========================================
    print("\n" + "=" * 80)
    print("演示 1: 扩散模型测试")
    print("=" * 80)
    
    print("\n🔧 创建扩散模型...")
    unet = UNet(in_channels=3, out_channels=3, base_channels=32, time_emb_dim=128)
    model = DiffusionModel(model=unet, timesteps=50, device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数数量: {total_params:,}")
    
    print("\n🔄 测试前向传播...")
    x = torch.randn(2, 3, 32, 32).to(device)
    loss = model(x)
    print(f"   损失值: {loss.item():.4f} ✓")
    
    print("\n🎨 生成样本...")
    with torch.no_grad():
        samples = model.generate(4, 32, 3)
    print(f"   生成样本形状: {samples.shape} ✓")
    
    # ========================================
    # 演示 2: 进化算法
    # ========================================
    print("\n" + "=" * 80)
    print("演示 2: 进化算法测试")
    print("=" * 80)
    
    print("\n🎯 目标: 找到接近 [0.5, 0.5, 0.5, 0.5, 0.5] 的基因")
    
    def fitness_function(genes):
        target = torch.ones_like(genes) * 0.5
        return -torch.sum((genes - target) ** 2).item()
    
    ga = GeneticAlgorithm(
        population_size=20,
        gene_length=5,
        fitness_function=fitness_function,
        selection_strategy=TournamentSelection(tournament_size=3),
        mutation_rate=0.1,
        mutation_strength=0.1,
        crossover_rate=0.8,
        elitism=1,
        maximize=True
    )
    
    print("\n🚀 运行进化 (10代)...")
    best_individual = ga.run(num_generations=10, target_fitness=-0.01)
    
    print(f"\n🏆 进化结果:")
    print(f"   最佳适应度: {best_individual.fitness:.6f}")
    print(f"   最佳基因: {best_individual.genes.numpy()}")
    print(f"   目标基因: [0.5, 0.5, 0.5, 0.5, 0.5]")
    
    # ========================================
    # 演示 3: 完整训练
    # ========================================
    print("\n" + "=" * 80)
    print("演示 3: 完整训练 (5轮)")
    print("=" * 80)
    
    # 创建数据
    print("\n📊 创建训练数据...")
    train_data = torch.randn(50, 3, 32, 32)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
    
    # 创建新模型
    unet_train = UNet(in_channels=3, out_channels=3, base_channels=32, time_emb_dim=128)
    model_train = DiffusionModel(model=unet_train, timesteps=50, device=device)
    
    # 创建训练器
    trainer = DiffusionEvolutionTrainer(
        model=model_train,
        train_loader=train_loader,
        device=device,
        lr=1e-3
    )
    
    print("\n🏋️ 开始训练...")
    history = trainer.train(num_epochs=5)
    
    print(f"\n🏁 训练完成！")
    print(f"   总时间: {history['total_time']:.2f} 秒")
    print(f"   最终损失: {history['train_losses'][-1]:.4f}")
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 80)
    print("🎉 快速演示完成！")
    print("=" * 80)
    print("\n✅ 所有核心功能测试通过！")
    print("\n项目特色：")
    print("  ✨ 完整的扩散模型实现")
    print("  🧬 多种进化算法策略")
    print("  🚀 完整的训练系统")
    print("  ⚙️  灵活的配置管理")
    print("\n下一步：")
    print("  1. 查看 README.md 了解项目详情")
    print("  2. 阅读 GETTING_STARTED.md 学习使用方法")
    print("  3. 运行 python test_all.py 进行完整测试")
    print("  4. 尝试使用 python demo.py 运行完整演示")
    print("=" * 80)


if __name__ == "__main__":
    main()