import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from typing import List, Dict, Optional
from molecule_encoder import MoleculeEncoder
from diffusion_model import ConditionalDiffusionModel
from solution_archive import SolutionArchive
from adaptive_scheduler import AdaptiveScheduler
from data_utils import compute_properties

class MoleculeOptimizationProblem(Problem):
    """
    分子优化问题定义
    """
    def __init__(self, encoder: MoleculeEncoder, target_logp: float):
        """
        初始化问题
        
        Args:
            encoder: 分子编码器
            target_logp: 目标logP值
        """
        super().__init__(n_var=128, n_obj=3, n_constr=0, xl=-3, xu=3)
        self.encoder = encoder
        self.target_logp = target_logp
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        评估函数
        
        Args:
            X: 决策变量矩阵
            out: 输出字典
        """
        n = len(X)
        objs = np.zeros((n, 3))
        
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
        
        out['F'] = objs

def initialize_population(problem: Problem, encoder: MoleculeEncoder, diffusion: ConditionalDiffusionModel, n_pop: int) -> List[Dict]:
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
    if n_dm > 0:
        # 从理想区间采样条件
        cond = np.random.uniform([0.8, 1.5, 100], [1.0, 2.5, 300], (n_dm, 3))
        # 生成潜在向量
        z_dm = diffusion.sample(cond)
        
        for i in range(n_dm):
            z = z_dm[i]
            smiles = encoder.decode(z)
            if smiles is not None:
                props = compute_properties(smiles)
                if props is not None:
                    objs = np.array([-props['qed'], abs(props['logp'] - problem.target_logp), props['mw']])
                    population.append({'z': z, 'objs': objs, 'smiles': smiles})
    
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
                    objs = np.array([-props['qed'], abs(props['logp'] - problem.target_logp), props['mw']])
                    population.append({'z': z, 'objs': objs, 'smiles': smiles})
    
    # 确保种群大小
    if len(population) < n_pop:
        # 重复现有个体
        while len(population) < n_pop:
            population.append(population[np.random.randint(len(population))])
    
    return population[:n_pop]

def generate_ga_offspring(problem: Problem, population: List[Dict], n_offspring: int) -> List[Dict]:
    """
    生成GA后代
    
    Args:
        problem: 优化问题
        population: 当前种群
        n_offspring: 后代数量
    
    Returns:
        GA后代
    """
    # 构建pymoo种群
    X = np.array([ind['z'] for ind in population])
    F = np.array([ind['objs'] for ind in population])
    
    pop = Population()
    for x, f in zip(X, F):
        ind = Individual(X=x, F=f)
        pop.append(ind)
    
    # 定义NSGA-II算法
    algorithm = NSGA2(
        pop_size=len(population),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # 生成后代
    offspring = algorithm._mating(pop, n_offspring)
    
    # 转换回自定义格式
    ga_offspring = []
    for ind in offspring:
        z = ind.X
        smiles = problem.encoder.decode(z)
        if smiles is not None:
            props = compute_properties(smiles)
            if props is not None:
                objs = np.array([-props['qed'], abs(props['logp'] - problem.target_logp), props['mw']])
                ga_offspring.append({'z': z, 'objs': objs, 'smiles': smiles})
    
    return ga_offspring

def generate_dm_offspring(problem: Problem, diffusion: ConditionalDiffusionModel, archive: SolutionArchive, n_offspring: int) -> List[Dict]:
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
    # 从存档中获取目标向量
    cond = archive.get_target_objectives(n_offspring)
    
    # 生成潜在向量
    z_dm = diffusion.sample(cond)
    
    # 转换为自定义格式
    dm_offspring = []
    for i in range(n_offspring):
        z = z_dm[i]
        smiles = problem.encoder.decode(z)
        if smiles is not None:
            props = compute_properties(smiles)
            if props is not None:
                objs = np.array([-props['qed'], abs(props['logp'] - problem.target_logp), props['mw']])
                dm_offspring.append({'z': z, 'objs': objs, 'smiles': smiles})
    
    return dm_offspring

def environmental_selection(combined: List[Dict], n_pop: int) -> List[Dict]:
    """
    环境选择
    
    Args:
        combined: 合并后的种群
        n_pop: 种群大小
    
    Returns:
        选择后的种群
    """
    # 构建pymoo种群
    X = np.array([ind['z'] for ind in combined])
    F = np.array([ind['objs'] for ind in combined])
    
    pop = Population()
    for x, f in zip(X, F):
        ind = Individual(X=x, F=f)
        pop.append(ind)
    
    # 定义NSGA-II算法进行选择
    algorithm = NSGA2(
        pop_size=n_pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # 执行选择
    algorithm.pop = pop
    algorithm.initialize()
    algorithm._initialize_advance()
    
    # 转换回自定义格式
    selected = []
    for ind in algorithm.pop:
        # 找到对应的原始个体
        for orig_ind in combined:
            if np.allclose(orig_ind['z'], ind.X):
                selected.append(orig_ind)
                break
    
    # 确保种群大小
    if len(selected) < n_pop:
        # 从剩余个体中随机选择
        remaining = [ind for ind in combined if ind not in selected]
        while len(selected) < n_pop and remaining:
            selected.append(remaining.pop(np.random.randint(len(remaining))))
    
    return selected[:n_pop]

def optimize(problem: MoleculeOptimizationProblem, encoder: MoleculeEncoder, diffusion: ConditionalDiffusionModel, 
             archive: SolutionArchive, scheduler: AdaptiveScheduler, n_gen: int = 200, 
             update_interval: int = 10, pop_size: int = 100) -> List[Dict]:
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
        最终种群
    """
    # 初始化种群
    population = initialize_population(problem, encoder, diffusion, pop_size)
    archive.add(population)
    
    for gen in range(n_gen):
        print(f"Generation {gen+1}/{n_gen}")
        
        # 计算后代数量
        total_offspring = pop_size
        
        # 生成GA后代
        ga_offspring_count = total_offspring - scheduler.get_dm_offspring_count(total_offspring, {})
        ga_offspring = generate_ga_offspring(problem, population, ga_offspring_count)
        
        # 生成DM后代
        dm_offspring_count = scheduler.get_dm_offspring_count(total_offspring, {})
        dm_offspring = generate_dm_offspring(problem, diffusion, archive, dm_offspring_count)
        
        # 合并后代
        offspring = ga_offspring + dm_offspring
        
        # 环境选择
        combined = population + offspring
        population = environmental_selection(combined, pop_size)
        
        # 更新存档
        archive.add(offspring)
        
        # 更新调度器性能
        pop_dict = {'objs': np.array([ind['objs'] for ind in population])}
        scheduler.update_performance(pop_dict)
        
        # 记录DM性能
        if dm_offspring:
            dm_survival = len([ind for ind in population if ind in dm_offspring]) / len(dm_offspring)
            scheduler.record_dm_performance(dm_survival)
        
        # 更新扩散模型
        if (gen + 1) % update_interval == 0:
            training_data = archive.get_training_data()
            if len(training_data['z']) >= 10:
                try:
                    print("Updating diffusion model...")
                    diffusion.train(training_data['z'], training_data['objs'], epochs=50)
                except Exception as e:
                    print(f"Diffusion model update failed: {e}")
                    print("Continuing with GA-only mode")
    
    return population
