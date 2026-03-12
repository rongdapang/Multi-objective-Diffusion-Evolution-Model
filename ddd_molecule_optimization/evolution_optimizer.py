"""
进化优化器模块 - 实现遗传算法和扩散模型的混合优化
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from molecule_encoder import MoleculeEncoder
from diffusion_model import ConditionalDiffusionModel
from solution_archive import SolutionArchive
from adaptive_scheduler import AdaptiveScheduler
from data_utils import compute_properties


class MoleculeOptimizationProblem:
    """
    分子优化问题定义
    """
    
    def __init__(self, encoder: MoleculeEncoder, target_logp: float,
                 n_var: int = 128, xl: float = -5, xu: float = 5):
        """
        初始化问题
        
        Args:
            encoder: 分子编码器
            target_logp: 目标logP值
            n_var: 决策变量维度
            xl: 下界
            xu: 上界
        """
        self.encoder = encoder
        self.target_logp = target_logp
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.n_obj = 3
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        评估函数
        
        Args:
            X: 决策变量矩阵 (shape: [n, n_var])
            
        Returns:
            目标值矩阵 (shape: [n, n_obj])
        """
        n = len(X)
        objs = np.zeros((n, self.n_obj))
        
        for i, z in enumerate(X):
            # 解码潜在向量
            smiles = self.encoder.decode(z)
            
            if smiles is None:
                # 解码失败，使用惩罚值
                objs[i] = [10.0, 10.0, 10000.0]
            else:
                # 计算分子性质
                props = compute_properties(smiles)
                if props is None:
                    # 分子无效，使用惩罚值
                    objs[i] = [10.0, 10.0, 10000.0]
                else:
                    # 计算目标值
                    objs[i, 0] = -props['qed']  # 最大化QED
                    objs[i, 1] = abs(props['logp'] - self.target_logp)  # 最小化logP偏差
                    objs[i, 2] = props['mw']  # 最小化分子量
        
        return objs


def sbx_crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  eta: float = 15.0, prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟二进制交叉(SBX)
    
    Args:
        parent1: 父代1
        parent2: 父代2
        eta: 分布指数
        prob: 交叉概率
        
    Returns:
        两个子代
    """
    if np.random.random() > prob:
        return parent1.copy(), parent2.copy()
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for i in range(len(parent1)):
        if np.random.random() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                if parent1[i] < parent2[i]:
                    y1, y2 = parent1[i], parent2[i]
                else:
                    y1, y2 = parent2[i], parent1[i]
                
                beta = 1.0 + (2.0 * (y1 - (-5)) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta + 1.0))
                
                rand = np.random.random()
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                
                c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                
                beta = 1.0 + (2.0 * (5 - y2) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta + 1.0))
                
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                
                c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                
                c1 = np.clip(c1, -5, 5)
                c2 = np.clip(c2, -5, 5)
                
                if np.random.random() <= 0.5:
                    child1[i] = c2
                    child2[i] = c1
                else:
                    child1[i] = c1
                    child2[i] = c2
    
    return child1, child2


def polynomial_mutation(x: np.ndarray, eta: float = 20.0, 
                        prob: float = 0.1, xl: float = -5, xu: float = 5) -> np.ndarray:
    """
    多项式变异
    
    Args:
        x: 个体
        eta: 分布指数
        prob: 变异概率
        xl: 下界
        xu: 上界
        
    Returns:
        变异后的个体
    """
    y = x.copy()
    
    for i in range(len(x)):
        if np.random.random() <= prob:
            delta1 = (y[i] - xl) / (xu - xl)
            delta2 = (xu - y[i]) / (xu - xl)
            
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1.0)
            
            if rand <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                delta_q = 1.0 - val ** mut_pow
            
            y[i] = y[i] + delta_q * (xu - xl)
            y[i] = np.clip(y[i], xl, xu)
    
    return y


def non_dominated_sort(objs: np.ndarray) -> List[List[int]]:
    """
    非支配排序
    
    Args:
        objs: 目标值矩阵
        
    Returns:
        前沿列表
    """
    n = len(objs)
    S = [[] for _ in range(n)]
    n_dominated = [0] * n
    fronts = [[]]
    
    for p in range(n):
        for q in range(n):
            if p != q:
                if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                    S[p].append(q)
                elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                    n_dominated[p] += 1
        
        if n_dominated[p] == 0:
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n_dominated[q] -= 1
                if n_dominated[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    
    return fronts[:-1]


def crowding_distance(objs: np.ndarray) -> np.ndarray:
    """
    计算拥挤度距离
    
    Args:
        objs: 目标值矩阵
        
    Returns:
        拥挤度距离数组
    """
    n, m = objs.shape
    distances = np.zeros(n)
    
    if n <= 2:
        return distances
    
    for i in range(m):
        sorted_indices = np.argsort(objs[:, i])
        sorted_objs = objs[sorted_indices, i]
        
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        obj_range = sorted_objs[-1] - sorted_objs[0]
        if obj_range > 0:
            for j in range(1, n - 1):
                distances[sorted_indices[j]] += (
                    sorted_objs[j + 1] - sorted_objs[j - 1]
                ) / obj_range
    
    return distances


def environmental_selection(population: List[dict], n_pop: int) -> List[dict]:
    """
    环境选择
    
    Args:
        population: 种群
        n_pop: 种群大小
        
    Returns:
        选择后的种群
    """
    if len(population) <= n_pop:
        return population
    
    # 提取目标值
    objs = np.array([ind['objs'] for ind in population])
    
    # 非支配排序
    fronts = non_dominated_sort(objs)
    
    # 选择个体
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n_pop:
            selected.extend(front)
        else:
            # 计算拥挤度距离
            front_objs = objs[front]
            distances = crowding_distance(front_objs)
            
            # 按拥挤度距离排序
            sorted_indices = np.argsort(distances)[::-1]
            remaining = n_pop - len(selected)
            selected.extend([front[i] for i in sorted_indices[:remaining]])
            break
    
    return [population[i] for i in selected]


def initialize_population(problem: MoleculeOptimizationProblem, 
                          encoder: MoleculeEncoder,
                          diffusion: ConditionalDiffusionModel, 
                          n_pop: int) -> List[Dict]:
    """
    初始化种群
    
    Args:
        problem: 优化问题
        encoder: 分子编码器
        diffusion: 扩散模型
        n_pop: 种群大小
        
    Returns:
        初始种群
    """
    population = []
    
    # 使用扩散模型生成一部分个体
    n_dm = n_pop // 2
    if n_dm > 0 and diffusion.is_trained:
        try:
            # 从理想区间采样条件
            cond = np.random.uniform([0.8, 0.5, 100], [1.0, 1.5, 300], (n_dm, 3))
            # 生成潜在向量
            z_dm = diffusion.sample(cond)
            
            for i in range(n_dm):
                z = z_dm[i]
                smiles = encoder.decode(z)
                if smiles is not None:
                    props = compute_properties(smiles)
                    if props is not None:
                        objs = np.array([
                            -props['qed'], 
                            abs(props['logp'] - problem.target_logp), 
                            props['mw']
                        ])
                        population.append({'z': z, 'objs': objs, 'smiles': smiles})
        except Exception as e:
            print(f"扩散模型初始化失败: {e}")
    
    # 随机采样补充种群
    n_random = n_pop - len(population)
    if n_random > 0:
        X = np.random.uniform(problem.xl, problem.xu, (n_random, problem.n_var))
        
        for i in range(n_random):
            z = X[i]
            smiles = encoder.decode(z)
            if smiles is not None:
                props = compute_properties(smiles)
                if props is not None:
                    objs = np.array([
                        -props['qed'], 
                        abs(props['logp'] - problem.target_logp), 
                        props['mw']
                    ])
                    population.append({'z': z, 'objs': objs, 'smiles': smiles})
    
    # 确保种群大小
    while len(population) < n_pop:
        if len(population) > 0:
            population.append(population[np.random.randint(len(population))].copy())
        else:
            # 如果种群为空，创建随机个体
            z = np.random.uniform(problem.xl, problem.xu, problem.n_var)
            population.append({'z': z, 'objs': np.array([10.0, 10.0, 10000.0]), 'smiles': None})
    
    return population[:n_pop]


def generate_ga_offspring(problem: MoleculeOptimizationProblem, 
                          population: List[Dict], 
                          n_offspring: int) -> List[Dict]:
    """
    生成GA后代
    
    Args:
        problem: 优化问题
        population: 当前种群
        n_offspring: 后代数量
        
    Returns:
        GA后代
    """
    ga_offspring = []
    n_parents = len(population)
    
    # 锦标赛选择
    tournament_size = 3
    
    while len(ga_offspring) < n_offspring:
        # 选择两个父代
        candidates1 = np.random.choice(n_parents, tournament_size, replace=False)
        candidates2 = np.random.choice(n_parents, tournament_size, replace=False)
        
        # 选择最优父代
        objs1 = np.array([population[i]['objs'] for i in candidates1])
        objs2 = np.array([population[i]['objs'] for i in candidates2])
        
        # 计算适应度（目标值之和，越小越好）
        fitness1 = np.sum(objs1, axis=1)
        fitness2 = np.sum(objs2, axis=1)
        
        parent1_idx = candidates1[np.argmin(fitness1)]
        parent2_idx = candidates2[np.argmin(fitness2)]
        
        parent1 = population[parent1_idx]['z']
        parent2 = population[parent2_idx]['z']
        
        # 交叉
        child1, child2 = sbx_crossover(parent1, parent2)
        
        # 变异
        child1 = polynomial_mutation(child1, xl=problem.xl, xu=problem.xu)
        child2 = polynomial_mutation(child2, xl=problem.xl, xu=problem.xu)
        
        # 评估
        for child in [child1, child2]:
            if len(ga_offspring) >= n_offspring:
                break
            
            smiles = problem.encoder.decode(child)
            if smiles is not None:
                props = compute_properties(smiles)
                if props is not None:
                    objs = np.array([
                        -props['qed'], 
                        abs(props['logp'] - problem.target_logp), 
                        props['mw']
                    ])
                    ga_offspring.append({'z': child, 'objs': objs, 'smiles': smiles})
    
    return ga_offspring


def generate_dm_offspring(problem: MoleculeOptimizationProblem,
                          diffusion: ConditionalDiffusionModel, 
                          archive: SolutionArchive, 
                          n_offspring: int) -> List[Dict]:
    """
    生成DM后代
    
    Args:
        problem: 优化问题
        diffusion: 扩散模型
        archive: 解决方案存档
        n_offspring: 后代数量
        
    Returns:
        DM后代
    """
    if not diffusion.is_trained:
        return []
    
    dm_offspring = []
    
    try:
        # 从存档中获取目标向量
        cond = archive.get_target_objectives(n_offspring)
        
        # 生成潜在向量
        z_dm = diffusion.sample(cond)
        
        # 转换为自定义格式
        for i in range(n_offspring):
            z = z_dm[i]
            smiles = problem.encoder.decode(z)
            if smiles is not None:
                props = compute_properties(smiles)
                if props is not None:
                    objs = np.array([
                        -props['qed'], 
                        abs(props['logp'] - problem.target_logp), 
                        props['mw']
                    ])
                    dm_offspring.append({'z': z, 'objs': objs, 'smiles': smiles})
    except Exception as e:
        print(f"生成DM后代时出错: {e}")
    
    return dm_offspring


def optimize(problem: MoleculeOptimizationProblem, 
             encoder: MoleculeEncoder, 
             diffusion: ConditionalDiffusionModel, 
             archive: SolutionArchive, 
             scheduler: AdaptiveScheduler, 
             n_gen: int = 200, 
             update_interval: int = 10, 
             pop_size: int = 100) -> Tuple[List[Dict], List[float]]:
    """
    进化优化主函数
    
    Args:
        problem: 优化问题
        encoder: 分子编码器
        diffusion: 扩散模型
        archive: 解决方案存档
        scheduler: 自适应调度器
        n_gen: 进化代数
        update_interval: 模型更新间隔
        pop_size: 种群大小
        
    Returns:
        最终种群和每代最优目标值历史
    """
    # 初始化种群
    print("初始化种群...")
    population = initialize_population(problem, encoder, diffusion, pop_size)
    archive.add(population)
    
    # 记录历史
    history = []
    
    for gen in range(n_gen):
        # 计算后代数量
        total_offspring = pop_size
        
        # 获取扩散模型后代数量
        dm_offspring_count = scheduler.get_dm_offspring_count(total_offspring, {})
        ga_offspring_count = total_offspring - dm_offspring_count
        
        # 生成GA后代
        ga_offspring = generate_ga_offspring(problem, population, ga_offspring_count)
        
        # 生成DM后代
        dm_offspring = generate_dm_offspring(problem, diffusion, archive, dm_offspring_count)
        
        # 合并后代
        offspring = ga_offspring + dm_offspring
        
        # 更新存档
        if offspring:
            archive.add(offspring)
        
        # 环境选择
        combined = population + offspring
        population = environmental_selection(combined, pop_size)
        
        # 更新调度器性能
        pop_objs = np.array([ind['objs'] for ind in population])
        scheduler.update_performance({'objs': pop_objs})
        
        # 记录DM性能
        if dm_offspring:
            # 使用id()比较对象身份，避免字典中包含numpy数组时的比较问题
            dm_offspring_ids = {id(ind) for ind in dm_offspring}
            dm_survival = len([ind for ind in population if id(ind) in dm_offspring_ids]) / len(dm_offspring) if dm_offspring else 0
            scheduler.record_dm_performance(dm_survival)
        
        # 记录历史
        best_obj = np.min(pop_objs, axis=0)
        history.append(np.sum(best_obj))
        
        # 打印进度
        if (gen + 1) % 10 == 0 or gen == 0:
            avg_qed = -np.mean(pop_objs[:, 0])
            avg_logp_error = np.mean(pop_objs[:, 1])
            avg_mw = np.mean(pop_objs[:, 2])
            print(f"Generation {gen+1}/{n_gen}: "
                  f"Avg QED={avg_qed:.3f}, "
                  f"Avg |logP-target|={avg_logp_error:.3f}, "
                  f"Avg MW={avg_mw:.1f}, "
                  f"DM ratio={scheduler.current_ratio:.2f}")
        
        # 更新扩散模型
        if (gen + 1) % update_interval == 0 and diffusion.is_trained:
            training_data = archive.get_training_data()
            if len(training_data['z']) >= 20:
                try:
                    print(f"Updating diffusion model at generation {gen+1}...")
                    diffusion.train(training_data['z'], training_data['objs'], epochs=20)
                except Exception as e:
                    print(f"扩散模型更新失败: {e}")
    
    return population, history
