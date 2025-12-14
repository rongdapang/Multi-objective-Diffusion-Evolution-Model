#!/usr/bin/env python3
"""
扩散-进化模型完整测试脚本
测试所有核心功能是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.diffusion import DiffusionModel, UNet
from models.evolution import GeneticAlgorithm, TournamentSelection
from training.trainer import DiffusionEvolutionTrainer
from configs.config import ConfigManager
from utils.utils import set_seed, save_samples, plot_training_curves


def test_diffusion_model():
    """测试扩散模型"""
    print("=" * 60)
    print("测试扩散模型")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(42)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256)
    model = DiffusionModel(model=unet, timesteps=100, device=device)
    
    # 测试前向传播
    print("1. 测试前向传播...")
    x = torch.randn(4, 3, 32, 32).to(device)
    loss = model(x)
    print(f"   损失值: {loss.item():.4f} ✓")
    
    # 测试样本生成
    print("2. 测试样本生成...")
    samples = model.generate(4, 32, 3)
    print(f"   生成样本形状: {samples.shape} ✓")
    
    # 测试损失是否合理
    assert not torch.isnan(loss), "损失包含NaN"
    assert loss.item() > 0, "损失应该为正数"
    assert samples.shape == (4, 3, 32, 32), "生成样本形状不正确"
    
    print("   所有检查通过 ✓")
    
    return model


def test_evolution_algorithm():
    """测试进化算法"""
    print("\n" + "=" * 60)
    print("测试进化算法")
    print("=" * 60)
    
    # 定义适应度函数
    def fitness_function(genes):
        target = torch.ones_like(genes) * 0.5
        return -torch.sum((genes - target) ** 2).item()
    
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
    
    print("1. 运行进化...")
    best_individual = ga.run(num_generations=20, target_fitness=-0.01)
    
    print(f"   最佳适应度: {best_individual.fitness:.6f} ✓")
    print(f"   最佳基因: {best_individual.genes[:5].numpy()}... ✓")
    
    # 检查适应度是否合理
    assert best_individual.fitness < 0, "适应度应该为负数"
    assert len(best_individual.genes) == 10, "基因长度不正确"
    
    stats = ga.get_statistics()
    print(f"   最终统计: {stats['generation']} 代, 最佳适应度: {stats['best_fitness']:.6f} ✓")
    
    print("   所有检查通过 ✓")
    
    return ga


def test_trainer():
    """测试训练器"""
    print("\n" + "=" * 60)
    print("测试训练器")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(42)
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建虚拟数据
    train_data = torch.randn(50, 3, 32, 32)
    val_data = torch.randn(20, 3, 32, 32)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=False)
    
    # 创建模型
    unet = UNet(in_channels=3, out_channels=3, base_channels=32, time_emb_dim=128)
    model = DiffusionModel(model=unet, timesteps=50, device=device)
    
    # 创建训练器
    trainer = DiffusionEvolutionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        save_dir="./test_checkpoints",
        log_dir="./test_logs"
    )
    
    print("1. 开始训练...")
    history = trainer.train(
        num_epochs=10,
        save_every=5,
        generate_every=10,
        evolution_interval=20,
        evolution_generations=5
    )
    
    print(f"   训练完成！总时间: {history['total_time']:.2f}秒 ✓")
    print(f"   最终训练损失: {history['train_losses'][-1]:.4f} ✓")
    print(f"   最佳验证损失: {history['best_val_loss']:.4f} ✓")
    
    # 检查训练历史
    assert len(history['train_losses']) == 10, "训练历史长度不正确"
    assert all(not torch.isnan(torch.tensor(loss)) for loss in history['train_losses']), "损失包含NaN"
    
    print("   所有检查通过 ✓")
    
    return trainer


def test_utils():
    """测试工具函数"""
    print("\n" + "=" * 60)
    print("测试工具函数")
    print("=" * 60)
    
    # 测试样本保存
    print("1. 测试样本保存...")
    samples = torch.randn(8, 3, 32, 32)
    save_path = "./test_samples.png"
    save_samples(samples, save_path, nrow=4)
    
    if os.path.exists(save_path):
        print(f"   样本已保存到 {save_path} ✓")
        os.remove(save_path)  # 清理
    else:
        print("   样本保存失败 ✗")
        return False
    
    # 测试随机种子设置
    print("2. 测试随机种子...")
    set_seed(42)
    a1 = torch.randn(5)
    set_seed(42)
    a2 = torch.randn(5)
    
    if torch.allclose(a1, a2):
        print("   随机种子设置正确 ✓")
    else:
        print("   随机种子设置失败 ✗")
        return False
    
    print("   所有检查通过 ✓")
    
    return True


def test_config():
    """测试配置系统"""
    print("\n" + "=" * 60)
    print("测试配置系统")
    print("=" * 60)
    
    # 创建默认配置
    config_manager = ConfigManager()
    
    print("1. 创建默认配置...")
    print(f"   模型基础通道: {config_manager.model_config.base_channels} ✓")
    print(f"   训练轮数: {config_manager.training_config.num_epochs} ✓")
    print(f"   进化种群大小: {config_manager.evolution_config.population_size} ✓")
    
    # 保存和加载配置
    print("2. 测试配置保存和加载...")
    config_path = "./test_config.json"
    config_manager.save_config(config_path)
    
    if os.path.exists(config_path):
        print(f"   配置已保存到 {config_path} ✓")
        
        # 加载配置
        new_config = ConfigManager()
        new_config.load_config(config_path)
        
        if new_config.model_config.base_channels == config_manager.model_config.base_channels:
            print("   配置加载成功 ✓")
        else:
            print("   配置加载失败 ✗")
            return False
        
        os.remove(config_path)  # 清理
    else:
        print("   配置保存失败 ✗")
        return False
    
    print("   所有检查通过 ✓")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("扩散-进化模型完整测试")
    print("=" * 60)
    
    tests = [
        ("扩散模型", test_diffusion_model),
        ("进化算法", test_evolution_algorithm),
        ("训练器", test_trainer),
        ("工具函数", test_utils),
        ("配置系统", test_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, result))
            print(f"\n{test_name} 测试: ✓ 通过")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n{test_name} 测试: ✗ 失败 - {str(e)}")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n总共 {total} 个测试，{passed} 个通过，{total - passed} 个失败\n")
    
    for test_name, success, result in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("🎉 所有测试通过！项目运行正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查代码。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)