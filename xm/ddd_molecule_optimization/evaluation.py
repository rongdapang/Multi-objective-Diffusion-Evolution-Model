import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict
from solution_archive import SolutionArchive

def evaluate_pareto_front(population: List[Dict], archive: SolutionArchive, ref_point: np.ndarray, save_path: str = 'pareto_front.csv') -> float:
    """
    评估Pareto前沿
    
    Args:
        population: 当前种群
        archive: 解决方案存档
        ref_point: 参考点
        save_path: 保存路径
    
    Returns:
        超体积值
    """
    # 合并种群和存档中的解
    all_solutions = population + archive.solutions
    
    # 去重
    unique_solutions = []
    seen_smiles = set()
    
    for sol in all_solutions:
        if sol['smiles'] not in seen_smiles and sol['smiles'] is not None:
            seen_smiles.add(sol['smiles'])
            unique_solutions.append(sol)
    
    # 计算Pareto前沿
    pareto_front = []
    for i, sol_i in enumerate(unique_solutions):
        dominated = False
        for j, sol_j in enumerate(unique_solutions):
            if i != j:
                # 检查sol_i是否被sol_j支配
                if np.all(sol_j['objs'] <= sol_i['objs']) and np.any(sol_j['objs'] < sol_i['objs']):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(sol_i)
    
    # 计算超体积
    objs = np.array([sol['objs'] for sol in pareto_front])
    hv = archive.calculate_hypervolume(objs, ref_point)
    
    # 保存Pareto前沿
    data = []
    for sol in pareto_front:
        props = {
            'smiles': sol['smiles'],
            'qed': -sol['objs'][0],  # 转换回原始值
            'logp': sol['objs'][1],  # 这里需要注意，因为objs[1]是abs(logp - target_logp)
            'mw': sol['objs'][2]
        }
        data.append(props)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Pareto前沿已保存到 {save_path}")
    print(f"超体积值: {hv:.4f}")
    
    return hv

def plot_pareto_front(objectives: np.ndarray, save_path: str = 'pareto_front.png'):
    """
    绘制Pareto前沿
    
    Args:
        objectives: 目标值矩阵
        save_path: 保存路径
    """
    if objectives.shape[1] == 3:
        # 3D散点图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='b', marker='o')
        
        ax.set_xlabel('f1 (-QED)')
        ax.set_ylabel('f2 (|logP - target|)')
        ax.set_zlabel('f3 (MW)')
        ax.set_title('Pareto Front')
    else:
        # 2D散点图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        ax.scatter(objectives[:, 0], objectives[:, 1], c='b', marker='o')
        
        ax.set_xlabel('f1 (-QED)')
        ax.set_ylabel('f2 (|logP - target|)')
        ax.set_title('Pareto Front')
    
    plt.savefig(save_path)
    print(f"Pareto前沿图已保存到 {save_path}")
    plt.close()
