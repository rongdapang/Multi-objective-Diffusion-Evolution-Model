#!/usr/bin/env python3
"""
扩散-进化模型实验运行脚本
支持多种数据集和配置
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CelebA, MNIST

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import DiffusionModel, UNet
from models.evolution import GeneticAlgorithm, TournamentSelection, DiffusionEvolutionOptimizer
from training.trainer import DiffusionEvolutionTrainer
from configs.config import ConfigManager, create_config_for_cifar10, create_config_for_custom_dataset
from utils.utils import setup_logging, set_seed, print_model_info


def load_dataset(dataset_name: str, data_path: str, image_size: int = 32, batch_size: int = 32):
    """加载数据集"""
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
    ])
    
    if dataset_name.lower() == "cifar10":
        train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
        val_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == "mnist":
        # 对于MNIST，转换为3通道
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
        val_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == "celeba":
        train_dataset = CelebA(root=data_path, split='train', download=True, transform=transform)
        val_dataset = CelebA(root=data_path, split='valid', download=True, transform=transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config_manager: ConfigManager):
    """创建模型"""
    model_config = config_manager.model_config
    
    # 创建UNet
    unet = UNet(
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        base_channels=model_config.base_channels,
        time_emb_dim=model_config.time_emb_dim
    )
    
    # 创建扩散模型
    diffusion_model = DiffusionModel(
        model=unet,
        timesteps=model_config.timesteps,
        beta_start=model_config.beta_start,
        beta_end=model_config.beta_end,
        device=model_config.device
    )
    
    return diffusion_model


def run_diffusion_training(config_manager: ConfigManager):
    """运行扩散模型训练"""
    print("Starting Diffusion Model Training...")
    
    # 配置
    model_config = config_manager.model_config
    training_config = config_manager.training_config
    experiment_config = config_manager.experiment_config
    
    # 设置随机种子
    set_seed(experiment_config.seed)
    
    # 设置日志
    logger = setup_logging(training_config.log_dir, experiment_config.log_level)
    
    # 加载数据
    logger.info(f"Loading dataset: {experiment_config.dataset_name}")
    train_loader, val_loader = load_dataset(
        experiment_config.dataset_name,
        experiment_config.data_path,
        training_config.image_size,
        training_config.batch_size
    )
    
    # 创建模型
    logger.info("Creating model...")
    model = create_model(config_manager)
    model.to(model_config.device)
    
    # 打印模型信息
    print_model_info(model.model, "UNet")
    
    # 创建训练器
    trainer = DiffusionEvolutionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=model_config.device,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        save_dir=training_config.save_dir,
        log_dir=training_config.log_dir
    )
    
    # 开始训练
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=training_config.num_epochs,
        save_every=training_config.save_every,
        generate_every=training_config.generate_every,
        evolution_interval=training_config.evolution_interval,
        evolution_generations=training_config.evolution_generations
    )
    
    # 保存最终模型
    final_model_path = os.path.join(training_config.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # 保存配置
    config_path = os.path.join(training_config.log_dir, 'config.json')
    config_manager.save_config(config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    return history


def run_evolution_only_test():
    """运行进化算法独立测试"""
    print("Running Evolution Algorithm Test...")
    
    # 简单的适应度函数
    def fitness_function(genes):
        # 最大化函数 f(x) = -sum((x - target)^2)
        target = torch.ones_like(genes) * 0.5
        return -torch.sum((genes - target) ** 2).item()
    
    # 创建遗传算法
    ga = GeneticAlgorithm(
        population_size=50,
        gene_length=10,
        fitness_function=fitness_function,
        selection_strategy=TournamentSelection(tournament_size=3),
        mutation_rate=0.1,
        mutation_strength=0.1,
        crossover_rate=0.8,
        elitism=2,
        maximize=True
    )
    
    # 运行进化
    print("Starting evolution...")
    best_individual = ga.run(num_generations=100, target_fitness=-0.01)
    
    if best_individual:
        print(f"Best fitness: {best_individual.fitness}")
        print(f"Best genes: {best_individual.genes}")
    
    # 打印统计信息
    stats = ga.get_statistics()
    print(f"Final statistics: {stats}")


def run_quick_test():
    """快速测试所有组件"""
    print("Running Quick Test...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单模型
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,  # 使用较小的模型
        time_emb_dim=256
    )
    
    diffusion_model = DiffusionModel(
        model=model,
        timesteps=100,  # 减少时间步数
        device=device
    )
    
    # 测试前向传播
    print("Testing forward pass...")
    x = torch.randn(4, 3, 32, 32).to(device)
    loss = diffusion_model(x)
    print(f"Loss: {loss.item():.4f}")
    
    # 测试样本生成
    print("Testing sample generation...")
    samples = diffusion_model.generate(4, 32, 3)
    print(f"Generated samples shape: {samples.shape}")
    
    print("Quick test completed successfully!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Diffusion-Evolution Model Experiments')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evolution_test', 'quick_test'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist', 'celeba', 'custom'])
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIFFUSION-EVOLUTION MODEL EXPERIMENTS")
    print("=" * 60)
    
    if args.mode == 'train':
        # 创建配置
        if args.config and os.path.exists(args.config):
            config_manager = ConfigManager()
            config_manager.load_config(args.config)
        elif args.dataset == 'cifar10':
            config_manager = create_config_for_cifar10()
        else:
            config_manager = create_config_for_custom_dataset()
        
        # 更新配置
        config_manager.training_config.num_epochs = args.epochs
        config_manager.training_config.batch_size = args.batch_size
        config_manager.training_config.learning_rate = args.lr
        config_manager.experiment_config.seed = args.seed
        config_manager.training_config.save_dir = args.save_dir
        config_manager.training_config.log_dir = args.log_dir
        
        # 打印配置
        config_manager.print_config()
        
        # 运行训练
        history = run_diffusion_training(config_manager)
        
    elif args.mode == 'evolution_test':
        run_evolution_only_test()
        
    elif args.mode == 'quick_test':
        run_quick_test()
    
    print("=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # 如果没有命令行参数，运行快速测试
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick test...")
        run_quick_test()
    else:
        main()