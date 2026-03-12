"""
DDD算法用于分子多目标优化 - 主程序

本程序实现了基于DDD（Diffusion-driven Design）算法的分子多目标优化系统，
用于生成具有特定性质的分子。

优化目标：
1. 最大化QED（类药性）
2. 最小化logP与目标值的偏差
3. 最小化分子量
"""

import argparse
import os
import sys
import numpy as np
import torch

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 导入自定义模块
from data_utils import load_qm9_data
from molecule_encoder import MoleculeEncoder
from diffusion_model import ConditionalDiffusionModel
from solution_archive import SolutionArchive
from adaptive_scheduler import AdaptiveScheduler
from evolution_optimizer import MoleculeOptimizationProblem, optimize
from evaluation import (
    evaluate_pareto_front, 
    plot_pareto_front, 
    plot_convergence_history,
    visualize_molecules,
    save_results_to_csv,
    print_summary
)


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='DDD算法用于分子多目标优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python main.py
  
  # 指定数据路径和参数
  python main.py --data_path data/qm9.csv --num_samples 10000 --target_logp 2.5
  
  # 调整进化参数
  python main.py --pop_size 150 --n_gen 300 --dm_epochs 150
  
  # 使用CPU运行
  python main.py --device cpu
        """
    )
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='qm9.csv',
                        help='QM9数据路径 (默认: qm9.csv)')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='使用样本数 (默认: 5000)')
    parser.add_argument('--target_logp', type=float, default=2.0,
                        help='目标logP值 (默认: 2.0)')
    
    # 进化参数
    parser.add_argument('--pop_size', type=int, default=100,
                        help='种群大小 (默认: 100)')
    parser.add_argument('--n_gen', type=int, default=200,
                        help='进化代数 (默认: 200)')
    parser.add_argument('--update_interval', type=int, default=10,
                        help='模型更新间隔 (默认: 10)')
    
    # 扩散模型参数
    parser.add_argument('--dm_epochs', type=int, default=100,
                        help='扩散模型初始训练轮数 (默认: 100)')
    parser.add_argument('--dm_steps', type=int, default=50,
                        help='扩散步数 (默认: 50)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='潜在空间维度 (默认: 128)')
    
    # 调度器参数
    parser.add_argument('--dm_ratio', type=float, default=0.4,
                        help='基础扩散模型比例 (默认: 0.4)')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备类型: cuda 或 cpu (默认: cuda)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='结果保存目录 (默认: results)')
    parser.add_argument('--skip_dm', action='store_true',
                        help='跳过扩散模型训练，仅使用GA')
    
    return parser.parse_args()


def check_dependencies():
    """
    检查必要的依赖是否已安装
    """
    missing = []
    
    try:
        import rdkit
    except ImportError:
        missing.append('rdkit')
    
    try:
        import torch
    except ImportError:
        missing.append('torch')
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')
    
    try:
        import pandas
    except ImportError:
        missing.append('pandas')
    
    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')
    
    if missing:
        print("错误: 缺少以下依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n请安装依赖:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 检查依赖
    check_dependencies()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*60)
    print("DDD分子多目标优化")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"目标logP: {args.target_logp}")
    print(f"种群大小: {args.pop_size}")
    print(f"进化代数: {args.n_gen}")
    print(f"设备: {args.device}")
    print("="*60)
    
    # 加载数据
    print("\n[1/6] 加载数据...")
    try:
        smiles_list, properties_df = load_qm9_data(args.num_samples, args.data_path)
        print(f"成功加载 {len(smiles_list)} 个有效分子")
    except Exception as e:
        print(f"加载数据失败: {e}")
        sys.exit(1)
    
    # 初始化编码器
    print("\n[2/6] 初始化编码器...")
    try:
        encoder = MoleculeEncoder(latent_dim=args.latent_dim, method='pca')
        encoder.fit(smiles_list)
        print(f"编码器拟合完成，潜在维度: {args.latent_dim}")
    except Exception as e:
        print(f"初始化编码器失败: {e}")
        sys.exit(1)
    
    # 编码训练集
    print("\n[3/6] 编码训练集...")
    try:
        z_train = encoder.encode_batch(smiles_list)
        cond_train = properties_df[['qed', 'logp', 'mw']].values
        print(f"训练数据: {len(z_train)} 个样本")
    except Exception as e:
        print(f"编码训练集失败: {e}")
        sys.exit(1)
    
    # 初始化并训练条件扩散模型
    print("\n[4/6] 初始化扩散模型...")
    diffusion = ConditionalDiffusionModel(
        z_dim=args.latent_dim,
        cond_dim=3,
        num_diffusion_steps=args.dm_steps,
        device=args.device
    )
    
    if not args.skip_dm:
        try:
            print("训练扩散模型...")
            losses = diffusion.train(z_train, cond_train, epochs=args.dm_epochs)
            print(f"扩散模型训练完成，最终损失: {losses[-1]:.6f}")
        except Exception as e:
            print(f"扩散模型训练失败: {e}")
            print("将使用GA-only模式继续")
    else:
        print("跳过扩散模型训练（GA-only模式）")
    
    # 初始化存档和调度器
    print("\n[5/6] 初始化存档和调度器...")
    archive = SolutionArchive(max_size=1000)
    scheduler = AdaptiveScheduler(
        base_dm_ratio=args.dm_ratio,
        min_ratio=0.1,
        max_ratio=0.7
    )
    
    # 定义优化问题
    problem = MoleculeOptimizationProblem(
        encoder=encoder,
        target_logp=args.target_logp,
        n_var=args.latent_dim
    )
    
    # 运行优化
    print("\n[6/6] 开始优化...")
    print("-"*60)
    try:
        population, history = optimize(
            problem=problem,
            encoder=encoder,
            diffusion=diffusion,
            archive=archive,
            scheduler=scheduler,
            n_gen=args.n_gen,
            update_interval=args.update_interval,
            pop_size=args.pop_size
        )
        print("-"*60)
        print("优化完成!")
    except Exception as e:
        print(f"优化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 评估结果
    print("\n评估结果...")
    results = evaluate_pareto_front(population, args.target_logp)
    
    # 打印摘要
    print_summary(results)
    
    # 保存结果
    print("\n保存结果...")
    
    # 绘制帕累托前沿
    try:
        plot_pareto_front(population, args.target_logp,
                          save_path=os.path.join(args.results_dir, 'pareto_front.png'))
    except Exception as e:
        print(f"绘制帕累托前沿失败: {e}")
    
    # 绘制收敛历史
    try:
        plot_convergence_history(history,
                                 save_path=os.path.join(args.results_dir, 'convergence.png'))
    except Exception as e:
        print(f"绘制收敛历史失败: {e}")
    
    # 可视化分子
    try:
        visualize_molecules(results['pareto_solutions'],
                            save_path=os.path.join(args.results_dir, 'molecules.png'))
    except Exception as e:
        print(f"可视化分子失败: {e}")
    
    # 保存CSV
    try:
        save_results_to_csv(population, args.target_logp,
                            save_path=os.path.join(args.results_dir, 'optimized_molecules.csv'))
    except Exception as e:
        print(f"保存CSV失败: {e}")
    
    print("\n" + "="*60)
    print("所有结果已保存到:", args.results_dir)
    print("="*60)


if __name__ == '__main__':
    main()
