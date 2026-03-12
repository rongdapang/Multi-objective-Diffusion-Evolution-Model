import argparse
import os
import numpy as np
import torch
from data_utils import load_qm9_data
from molecule_encoder import MoleculeEncoder
from diffusion_model import ConditionalDiffusionModel
from solution_archive import SolutionArchive
from adaptive_scheduler import AdaptiveScheduler
from evolution_optimizer import MoleculeOptimizationProblem, optimize
from evaluation import evaluate_pareto_front, plot_pareto_front

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DDD算法用于分子多目标优化')
    parser.add_argument('--data_path', type=str, default='qm9.csv', help='QM9数据路径')
    parser.add_argument('--num_samples', type=int, default=5000, help='使用样本数')
    parser.add_argument('--target_logp', type=float, default=2.0, help='目标logP值')
    parser.add_argument('--pop_size', type=int, default=100, help='种群大小')
    parser.add_argument('--n_gen', type=int, default=200, help='进化代数')
    parser.add_argument('--dm_epochs', type=int, default=100, help='扩散模型初始训练轮数')
    parser.add_argument('--update_interval', type=int, default=10, help='模型更新间隔')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    args = parser.parse_args()
    
    # 固定随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    smiles_list, properties_df = load_qm9_data(args.num_samples, args.data_path)
    print(f"加载了 {len(smiles_list)} 个有效分子")
    
    # 初始化编码器
    print("初始化编码器...")
    encoder = MoleculeEncoder(latent_dim=128, method='pca')
    encoder.fit(smiles_list)
    
    # 编码训练集
    print("编码训练集...")
    z_train = encoder.encode_batch(smiles_list)
    cond_train = properties_df[['qed', 'logp', 'mw']].values
    
    # 初始化并训练条件扩散模型
    print("初始化并训练扩散模型...")
    diffusion = ConditionalDiffusionModel(z_dim=128, cond_dim=3, device=args.device)
    try:
        diffusion.train(z_train, cond_train, epochs=args.dm_epochs)
    except Exception as e:
        print(f"扩散模型训练失败: {e}")
        print("切换到GA-only模式")
    
    # 初始化存档、调度器
    print("初始化存档和调度器...")
    archive = SolutionArchive(max_size=200)
    scheduler = AdaptiveScheduler()
    
    # 定义优化问题
    print("定义优化问题...")
    problem = MoleculeOptimizationProblem(encoder, args.target_logp)
    
    # 运行进化优化
    print("运行进化优化...")
    population = optimize(
        problem=problem,
        encoder=encoder,
        diffusion=diffusion,
        archive=archive,
        scheduler=scheduler,
        n_gen=args.n_gen,
        update_interval=args.update_interval,
        pop_size=args.pop_size
    )
    
    # 评估并可视化结果
    print("评估并可视化结果...")
    ref_point = np.array([1.1, 1.1, 1000.0])  # 参考点
    hv = evaluate_pareto_front(population, archive, ref_point, 'results/pareto_front.csv')
    
    # 绘制Pareto前沿
    all_objs = np.array([sol['objs'] for sol in population + archive.solutions])
    plot_pareto_front(all_objs, 'results/pareto_front.png')
    
    print("优化完成！")
    print(f"超体积值: {hv:.4f}")
    print("结果已保存到 results 目录")

if __name__ == '__main__':
    main()
