"""
评估模块 - 用于评估和可视化优化结果
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import Draw
import os


def evaluate_pareto_front(population: List[dict], target_logp: float) -> Dict[str, any]:
    """
    评估帕累托前沿
    
    Args:
        population: 最终种群
        target_logp: 目标logP值
        
    Returns:
        评估结果字典
    """
    if not population:
        return {
            'num_solutions': 0,
            'avg_qed': 0.0,
            'avg_logp_error': 0.0,
            'avg_mw': 0.0,
            'best_qed': 0.0,
            'best_logp_error': float('inf'),
            'pareto_solutions': []
        }
    
    # 提取目标值
    objs = np.array([ind['objs'] for ind in population])
    
    # 计算非支配解
    is_pareto = np.ones(len(population), dtype=bool)
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    is_pareto[i] = False
                    break
    
    pareto_indices = np.where(is_pareto)[0]
    pareto_solutions = [population[i] for i in pareto_indices]
    pareto_objs = objs[pareto_indices]
    
    # 计算统计信息
    qed_values = -pareto_objs[:, 0]  # 转换为正值（最大化）
    logp_errors = pareto_objs[:, 1]
    mw_values = pareto_objs[:, 2]
    
    results = {
        'num_solutions': len(pareto_solutions),
        'num_total': len(population),
        'avg_qed': np.mean(qed_values),
        'std_qed': np.std(qed_values),
        'avg_logp_error': np.mean(logp_errors),
        'std_logp_error': np.std(logp_errors),
        'avg_mw': np.mean(mw_values),
        'std_mw': np.std(mw_values),
        'best_qed': np.max(qed_values),
        'best_logp_error': np.min(logp_errors),
        'best_mw': np.min(mw_values),
        'pareto_solutions': pareto_solutions
    }
    
    return results


def plot_pareto_front(population: List[dict], target_logp: float, 
                      save_path: str = 'results/pareto_front.png'):
    """
    绘制帕累托前沿
    
    Args:
        population: 最终种群
        target_logp: 目标logP值
        save_path: 保存路径
    """
    if not population:
        print("种群为空，无法绘制帕累托前沿")
        return
    
    # 提取目标值
    objs = np.array([ind['objs'] for ind in population])
    
    # 计算非支配解
    is_pareto = np.ones(len(population), dtype=bool)
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    is_pareto[i] = False
                    break
    
    pareto_indices = np.where(is_pareto)[0]
    pareto_objs = objs[pareto_indices]
    
    # 转换目标值
    qed_all = -objs[:, 0]
    logp_error_all = objs[:, 1]
    mw_all = objs[:, 2]
    
    qed_pareto = -pareto_objs[:, 0]
    logp_error_pareto = pareto_objs[:, 1]
    mw_pareto = pareto_objs[:, 2]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # QED vs logP Error
    ax1 = fig.add_subplot(131)
    ax1.scatter(qed_all, logp_error_all, c='lightgray', alpha=0.5, s=30, label='All solutions')
    ax1.scatter(qed_pareto, logp_error_pareto, c='red', s=50, label='Pareto front')
    ax1.set_xlabel('QED (Drug-likeness)', fontsize=12)
    ax1.set_ylabel(f'|logP - {target_logp}|', fontsize=12)
    ax1.set_title('QED vs logP Error', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QED vs MW
    ax2 = fig.add_subplot(132)
    ax2.scatter(qed_all, mw_all, c='lightgray', alpha=0.5, s=30, label='All solutions')
    ax2.scatter(qed_pareto, mw_pareto, c='red', s=50, label='Pareto front')
    ax2.set_xlabel('QED (Drug-likeness)', fontsize=12)
    ax2.set_ylabel('Molecular Weight', fontsize=12)
    ax2.set_title('QED vs Molecular Weight', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # logP Error vs MW
    ax3 = fig.add_subplot(133)
    ax3.scatter(logp_error_all, mw_all, c='lightgray', alpha=0.5, s=30, label='All solutions')
    ax3.scatter(logp_error_pareto, mw_pareto, c='red', s=50, label='Pareto front')
    ax3.set_xlabel(f'|logP - {target_logp}|', fontsize=12)
    ax3.set_ylabel('Molecular Weight', fontsize=12)
    ax3.set_title('logP Error vs Molecular Weight', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"帕累托前沿图已保存到 {save_path}")
    plt.close()


def plot_convergence_history(history: List[float], save_path: str = 'results/convergence.png'):
    """
    绘制收敛历史
    
    Args:
        history: 每代最优目标值历史
        save_path: 保存路径
    """
    if not history:
        print("历史为空，无法绘制收敛图")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Objective Value', fontsize=12)
    plt.title('Convergence History', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"收敛历史图已保存到 {save_path}")
    plt.close()


def visualize_molecules(solutions: List[dict], num_molecules: int = 9,
                        save_path: str = 'results/molecules.png'):
    """
    可视化分子结构
    
    Args:
        solutions: 解列表
        num_molecules: 要可视化的分子数量
        save_path: 保存路径
    """
    if not solutions:
        print("解列表为空，无法可视化分子")
        return
    
    # 选择QED最高的分子
    sorted_solutions = sorted(solutions, key=lambda x: -x['objs'][0])
    selected = sorted_solutions[:num_molecules]
    
    # 创建分子对象
    mols = []
    legends = []
    
    for sol in selected:
        smiles = sol['smiles']
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                qed = -sol['objs'][0]
                logp_error = sol['objs'][1]
                mw = sol['objs'][2]
                legends.append(f'QED={qed:.3f}\n|logP-target|={logp_error:.2f}\nMW={mw:.1f}')
    
    if not mols:
        print("没有有效的分子可以可视化")
        return
    
    # 绘制分子
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300),
                                legends=legends)
    img.save(save_path)
    print(f"分子可视化图已保存到 {save_path}")


def save_results_to_csv(population: List[dict], target_logp: float,
                        save_path: str = 'results/optimized_molecules.csv'):
    """
    保存结果到CSV文件
    
    Args:
        population: 最终种群
        target_logp: 目标logP值
        save_path: 保存路径
    """
    if not population:
        print("种群为空，无法保存结果")
        return
    
    import csv
    
    # 计算非支配解
    objs = np.array([ind['objs'] for ind in population])
    is_pareto = np.ones(len(population), dtype=bool)
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    is_pareto[i] = False
                    break
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES', 'QED', 'logP_Error', 'MW', 'Is_Pareto'])
        
        for i, ind in enumerate(population):
            smiles = ind['smiles'] if ind['smiles'] else ''
            qed = -ind['objs'][0]
            logp_error = ind['objs'][1]
            mw = ind['objs'][2]
            is_pareto_flag = 'Yes' if is_pareto[i] else 'No'
            writer.writerow([smiles, qed, logp_error, mw, is_pareto_flag])
    
    print(f"结果已保存到 {save_path}")


def print_summary(results: Dict[str, any]):
    """
    打印评估摘要
    
    Args:
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("优化结果摘要")
    print("="*60)
    print(f"帕累托解数量: {results['num_solutions']} / {results['num_total']}")
    print(f"\n平均 QED: {results['avg_qed']:.4f} (±{results['std_qed']:.4f})")
    print(f"平均 logP 误差: {results['avg_logp_error']:.4f} (±{results['std_logp_error']:.4f})")
    print(f"平均分子量: {results['avg_mw']:.2f} (±{results['std_mw']:.2f})")
    print(f"\n最佳 QED: {results['best_qed']:.4f}")
    print(f"最佳 logP 误差: {results['best_logp_error']:.4f}")
    print(f"最佳分子量: {results['best_mw']:.2f}")
    print("="*60)
