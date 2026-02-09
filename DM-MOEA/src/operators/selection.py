"""
选择操作模块

该模块实现了多目标进化算法中常用的选择操作，包括：
- 快速非支配排序（Fast Non-dominated Sorting）
- 拥挤距离计算（Crowding Distance Calculation）
- 环境选择（Environmental Selection）
- 锦标赛选择（Tournament Selection）

这些操作是NSGA-II等经典多目标进化算法的核心组件。
"""

import numpy as np


def fast_non_dominated_sort(population):
    """
    快速非支配排序算法
    
    该算法将种群中的个体按照非支配关系分成多个前沿（fronts）。
    第一前沿包含所有非支配解，第二前沿包含被第一前沿中的个体支配但互不支配的解，以此类推。
    
    时间复杂度：O(MN^2)，其中M是目标数量，N是种群大小
    
    参数:
        population (list): 个体列表，每个个体必须有dominates方法
        
    返回:
        list: 前沿列表，每个前沿是一个个体列表
    """
    fronts = [[]]
    
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = set()
        
        for q in population:
            if p == q:
                continue
            if p.dominates(q):
                p.dominated_solutions.add(q)
            elif q.dominates(p):
                p.domination_count += 1
        
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    
    return fronts[:-1]


def compute_crowding_distance(front):
    """
    计算拥挤距离
    
    拥挤距离用于衡量个体在目标空间中的拥挤程度。
    拥挤距离越大，表示个体周围的解越稀疏，多样性越好。
    边界解（在每个目标上最小或最大的解）的拥挤距离设为无穷大。
    
    参数:
        front (list): 同一前沿的个体列表
        
    返回:
        None: 直接修改个体的crowding_distance属性
    """
    if len(front) == 0:
        return
    
    n_obj = len(front[0].objectives)
    
    for ind in front:
        ind.crowding_distance = 0.0
    
    for m in range(n_obj):
        front.sort(key=lambda x: x.objectives[m])
        
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        
        if obj_max - obj_min < 1e-10:
            continue
        
        for i in range(1, len(front) - 1):
            distance = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / (obj_max - obj_min)
            front[i].crowding_distance += distance


def environmental_selection(population, N):
    """
    环境选择
    
    从合并后的种群中选择N个最优个体进入下一代。
    选择策略：
    1. 优先选择非支配等级低的个体
    2. 对于同一前沿的个体，优先选择拥挤距离大的个体
    
    参数:
        population (list): 合并后的种群（父代+子代）
        N (int): 需要选择的个体数量
        
    返回:
        list: 选中的N个个体
    """
    fronts = fast_non_dominated_sort(population)
    
    for front in fronts:
        compute_crowding_distance(front)
    
    next_pop = []
    i = 0
    while len(next_pop) + len(fronts[i]) <= N:
        next_pop.extend(fronts[i])
        i += 1
    
    if len(next_pop) < N:
        remaining = N - len(next_pop)
        fronts[i].sort(key=lambda x: x.crowding_distance, reverse=True)
        next_pop.extend(fronts[i][:remaining])
    
    return next_pop


def tournament_selection(population, tournament_size=2):
    """
    锦标赛选择
    
    从种群中随机选择tournament_size个个体，选择其中最优的个体。
    个体优劣比较标准：
    1. 非支配等级越小越好
    2. 如果等级相同，拥挤距离越大越好
    
    参数:
        population (list): 种群列表
        tournament_size (int): 锦标赛大小，默认为2
        
    返回:
        list: 选中的个体列表
    """
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(population, tournament_size, replace=False)
        best = min(candidates, key=lambda x: (x.rank, -x.crowding_distance))
        selected.append(best)
    return selected


def binary_tournament_selection(p, q):
    """
    二元锦标赛选择
    
    比较两个个体，返回较优的个体。
    
    参数:
        p (Individual): 第一个个体
        q (Individual): 第二个个体
        
    返回:
        Individual: 较优的个体
    """
    if p.rank < q.rank:
        return p
    elif p.rank > q.rank:
        return q
    else:
        return p if p.crowding_distance > q.crowding_distance else q
