"""
工具函数

包含数据归一化、反归一化、多样性计算等实用函数。
"""

import numpy as np
from typing import Tuple
from .core import Population


def normalize(x: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    将数据归一化到 [0, 1] 范围
    
    参数:
        x: 输入数据 (n_samples, n_features)
        lower_bound: 下界 (n_features,)
        upper_bound: 上界 (n_features,)
        
    返回:
        归一化后的数据
    """
    return (x - lower_bound) / (upper_bound - lower_bound)


def denormalize(x: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    将数据从 [0, 1] 范围反归一化
    
    参数:
        x: 归一化数据 (n_samples, n_features)
        lower_bound: 下界 (n_features,)
        upper_bound: 上界 (n_features,)
        
    返回:
        反归一化后的数据
    """
    return x * (upper_bound - lower_bound) + lower_bound


def calculate_diversity(population: Population) -> float:
    """
    计算种群多样性
    
    使用平均成对距离作为多样性度量
    
    参数:
        population: 种群
        
    返回:
        多样性度量值
    """
    if len(population) < 2:
        return 0.0
        
    decision_matrix = population.get_decision_matrix()
    n_individuals = len(population)
    
    # 计算成对距离
    total_distance = 0.0
    n_pairs = 0
    
    for i in range(n_individuals):
        for j in range(i + 1, n_individuals):
            distance = np.linalg.norm(decision_matrix[i] - decision_matrix[j])
            total_distance += distance
            n_pairs += 1
            
    return total_distance / n_pairs if n_pairs > 0 else 0.0


def calculate_hypervolume(objectives: np.ndarray, reference_point: Optional[np.ndarray] = None) -> float:
    """
    计算超体积 (Hypervolume)
    
    参数:
        objectives: 目标函数矩阵 (n_solutions, n_objectives)
        reference_point: 参考点，如果为None则自动计算
        
    返回:
        超体积值
    """
    n_solutions, n_objectives = objectives.shape
    
    if n_solutions == 0:
        return 0.0
        
    # 如果没有提供参考点，使用最大值
    if reference_point is None:
        reference_point = np.max(objectives, axis=0) * 1.1
        
    # 简化的超体积计算（适用于2-3个目标）
    if n_objectives == 2:
        return _calculate_hypervolume_2d(objectives, reference_point)
    elif n_objectives == 3:
        return _calculate_hypervolume_3d(objectives, reference_point)
    else:
        # 对于高维目标，使用蒙特卡洛估计
        return _calculate_hypervolume_monte_carlo(objectives, reference_point, n_samples=10000)


def _calculate_hypervolume_2d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """计算2维超体积"""
    # 按第一个目标排序
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_objectives = objectives[sorted_indices]
    
    # 计算超体积
    hv = 0.0
    prev_f1 = reference_point[0]
    
    for i in range(len(sorted_objectives)):
        f1, f2 = sorted_objectives[i]
        if f1 < prev_f1 and f2 < reference_point[1]:
            hv += (prev_f1 - f1) * (reference_point[1] - f2)
            prev_f1 = f1
            
    return hv


def _calculate_hypervolume_3d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """计算3维超体积"""
    # 按第一个目标排序
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_objectives = objectives[sorted_indices]
    
    hv = 0.0
    
    for i in range(len(sorted_objectives)):
        f1 = sorted_objectives[i, 0]
        if f1 >= reference_point[0]:
            continue
            
        # 计算剩余目标的2维超体积
        remaining_objectives = sorted_objectives[i:, 1:]
        remaining_reference = reference_point[1:]
        
        # 只考虑在f1维度上更好的解
        mask = remaining_objectives[:, 0] <= reference_point[1]
        mask = mask & (remaining_objectives[:, 1] <= reference_point[2])
        
        if np.any(mask):
            hv_2d = _calculate_hypervolume_2d(remaining_objectives[mask], remaining_reference)
            if i == 0:
                delta_f1 = reference_point[0] - f1
            else:
                delta_f1 = sorted_objectives[i-1, 0] - f1
                
            hv += delta_f1 * hv_2d
            
    return hv


def _calculate_hypervolume_monte_carlo(objectives: np.ndarray, reference_point: np.ndarray, 
                                      n_samples: int = 10000) -> float:
    """使用蒙特卡洛方法估计超体积"""
    n_solutions = len(objectives)
    n_objectives = objectives.shape[1]
    
    # 采样点
    samples = np.random.uniform(0, 1, (n_samples, n_objectives))
    
    # 缩放采样点到参考点
    samples = samples * reference_point
    
    # 计算被支配的采样点数量
    dominated_count = 0
    
    for sample in samples:
        # 检查是否有解支配这个采样点
        is_dominated = False
        for solution in objectives:
            if np.all(solution <= sample) and np.any(solution < sample):
                is_dominated = True
                break
                
        if is_dominated:
            dominated_count += 1
            
    # 估计超体积
    total_volume = np.prod(reference_point)
    hv = (dominated_count / n_samples) * total_volume
    
    return hv


def calculate_igd(population: Population, true_pareto_front: np.ndarray) -> float:
    """
    计算Inverted Generational Distance (IGD)
    
    参数:
        population: 种群
        true_pareto_front: 真实帕累托前沿
        
    返回:
        IGD值
    """
    if len(population) == 0 or len(true_pareto_front) == 0:
        return float('inf')
        
    objectives = population.get_objective_matrix()
    
    # 计算每个真实帕累托解到最近解的距离
    distances = []
    for pf_point in true_pareto_front:
        # 计算到所有解的欧氏距离
        dists = np.linalg.norm(objectives - pf_point, axis=1)
        min_dist = np.min(dists)
        distances.append(min_dist)
        
    return np.mean(distances)


def calculate_gd(population: Population, true_pareto_front: np.ndarray) -> float:
    """
    计算Generational Distance (GD)
    
    参数:
        population: 种群
        true_pareto_front: 真实帕累托前沿
        
    返回:
        GD值
    """
    if len(population) == 0 or len(true_pareto_front) == 0:
        return float('inf')
        
    objectives = population.get_objective_matrix()
    
    # 计算每个解到真实帕累托前沿的距离
    distances = []
    for solution in objectives:
        # 计算到所有帕累托点的欧氏距离
        dists = np.linalg.norm(true_pareto_front - solution, axis=1)
        min_dist = np.min(dists)
        distances.append(min_dist)
        
    return np.mean(distances)


def calculate_spread(population: Population) -> float:
    """
    计算解的分布广度
    
    参数:
        population: 种群
        
    返回:
        分布广度值
    """
    if len(population) < 2:
        return 0.0
        
    objectives = population.get_objective_matrix()
    n_objectives = objectives.shape[1]
    
    # 计算每个维度的范围
    spread = 0.0
    for m in range(n_objectives):
        obj_range = np.max(objectives[:, m]) - np.min(objectives[:, m])
        spread += obj_range
        
    return spread / n_objectives


def feasible_rate(population: Population) -> float:
    """
    计算可行解比例
    
    参数:
        population: 种群
        
    返回:
        可行解比例 [0, 1]
    """
    if len(population) == 0:
        return 0.0
        
    constraint_matrix = population.get_constraint_matrix()
    
    if constraint_matrix.size == 0:
        return 1.0  # 无约束问题，全部可行
        
    # 检查每个个体是否满足所有约束
    feasible_count = 0
    for i in range(len(population)):
        if np.all(constraint_matrix[i] <= 0):
            feasible_count += 1
            
    return feasible_count / len(population)


def print_population_info(population: Population, generation: int = 0):
    """
    打印种群信息
    
    参数:
        population: 种群
        generation: 当前代数
    """
    if len(population) == 0:
        print(f"第{generation}代: 空种群")
        return
        
    objectives = population.get_objective_matrix()
    
    print(f"\n第{generation}代种群信息:")
    print(f"  种群大小: {len(population)}")
    print(f"  目标函数范围:")
    
    for m in range(objectives.shape[1]):
        obj_min = np.min(objectives[:, m])
        obj_max = np.max(objectives[:, m])
        obj_mean = np.mean(objectives[:, m])
        print(f"    f{m+1}: [{obj_min:.4f}, {obj_max:.4f}] (均值: {obj_mean:.4f})")
        
    # 多样性信息
    diversity = calculate_diversity(population)
    print(f"  种群多样性: {diversity:.4f}")
    
    # 约束信息
    constraint_matrix = population.get_constraint_matrix()
    if constraint_matrix.size > 0:
        feasible_rate_val = feasible_rate(population)
        print(f"  可行解比例: {feasible_rate_val:.2%}")
        
    print()


def save_results(results: dict, filename: str):
    """
    保存优化结果
    
    参数:
        results: 优化结果字典
        filename: 保存文件名
    """
    import pickle
    
    # 提取可序列化的数据
    save_data = {
        'decision_vars': results['decision_vars'],
        'objectives': results['objectives'],
        'n_evaluations': results['n_evaluations'],
        'runtime': results['runtime'],
        'config': results['config'].__dict__ if hasattr(results['config'], '__dict__') else str(results['config'])
    }
    
    if 'constraints' in results and results['constraints'] is not None:
        save_data['constraints'] = results['constraints']
        
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"结果已保存到: {filename}")


def load_results(filename: str) -> dict:
    """
    加载优化结果
    
    参数:
        filename: 文件名
        
    返回:
        结果字典
    """
    import pickle
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
        
    return results