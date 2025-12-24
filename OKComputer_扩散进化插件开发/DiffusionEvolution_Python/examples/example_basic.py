"""
基础使用示例

展示如何使用 DiffusionEvolution 算法解决简单的多目标优化问题。
"""

import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_evolution import DiffusionEvolution, DiffusionEvolutionConfig
from diffusion_evolution.utils import print_population_info, calculate_hypervolume, save_results


class ZDT1:
    """ZDT1 测试问题"""
    
    def __init__(self):
        self.n_variables = 30
        self.n_objectives = 2
        self.lower_bound = np.zeros(self.n_variables)
        self.upper_bound = np.ones(self.n_variables)
        
    def evaluate(self, decision_vars):
        """
        评估决策变量
        
        参数:
            decision_vars: 决策变量矩阵 (n_individuals, n_variables)
            
        返回:
            字典，包含目标函数值
        """
        n_individuals = len(decision_vars)
        objectives = np.zeros((n_individuals, self.n_objectives))
        
        # 第一个目标函数
        objectives[:, 0] = decision_vars[:, 0]
        
        # 计算 g 函数
        g = 1 + 9 * np.mean(decision_vars[:, 1:], axis=1)
        
        # 第二个目标函数
        objectives[:, 1] = g * (1 - np.sqrt(decision_vars[:, 0] / g))
        
        return {'objectives': objectives}


class ZDT2:
    """ZDT2 测试问题"""
    
    def __init__(self):
        self.n_variables = 30
        self.n_objectives = 2
        self.lower_bound = np.zeros(self.n_variables)
        self.upper_bound = np.ones(self.n_variables)
        
    def evaluate(self, decision_vars):
        """评估决策变量"""
        n_individuals = len(decision_vars)
        objectives = np.zeros((n_individuals, self.n_objectives))
        
        # 第一个目标函数
        objectives[:, 0] = decision_vars[:, 0]
        
        # 计算 g 函数
        g = 1 + 9 * np.mean(decision_vars[:, 1:], axis=1)
        
        # 第二个目标函数（与ZDT1不同）
        objectives[:, 1] = g * (1 - (decision_vars[:, 0] / g) ** 2)
        
        return {'objectives': objectives}


class DTLZ2:
    """DTLZ2 测试问题（多目标）"""
    
    def __init__(self, n_objectives=3, n_variables=None):
        self.n_objectives = n_objectives
        self.k = n_objectives - 1  # 距离变量数量
        
        if n_variables is None:
            self.n_variables = n_objectives + 9
        else:
            self.n_variables = n_variables
            
        self.lower_bound = np.zeros(self.n_variables)
        self.upper_bound = np.ones(self.n_variables)
        
    def evaluate(self, decision_vars):
        """评估决策变量"""
        n_individuals = len(decision_vars)
        objectives = np.zeros((n_individuals, self.n_objectives))
        
        # 位置变量
        xm = decision_vars[:, self.n_objectives-1:]
        
        # 计算 g 函数
        g = np.sum((xm - 0.5) ** 2, axis=1)
        
        # 计算目标函数
        for i in range(self.n_objectives):
            if i == 0:
                f = (1 + g) * np.cos(decision_vars[:, 0] * np.pi / 2)
            elif i == self.n_objectives - 1:
                f = (1 + g) * np.sin(decision_vars[:, 0] * np.pi / 2)
                for j in range(1, self.n_objectives - 1):
                    f *= np.sin(decision_vars[:, j] * np.pi / 2)
            else:
                f = (1 + g) * np.cos(decision_vars[:, i] * np.pi / 2)
                for j in range(i + 1, self.n_objectives - 1):
                    f *= np.sin(decision_vars[:, j] * np.pi / 2)
                    
            objectives[:, i] = f
            
        return {'objectives': objectives}


def run_basic_example():
    """运行基础示例"""
    print("=" * 60)
    print("DiffusionEvolution 基础使用示例")
    print("=" * 60)
    
    # 示例1: 使用默认配置优化 ZDT1
    print("\n示例 1: 使用默认配置优化 ZDT1")
    print("-" * 40)
    
    problem = ZDT1()
    algorithm = DiffusionEvolution()
    
    print("开始优化 ZDT1...")
    results = algorithm.solve(problem, max_evaluations=1000, verbose=True)
    
    print(f"\n优化结果:")
    print(f"  评估次数: {results['n_evaluations']}")
    print(f"  运行时间: {results['runtime']:.2f}秒")
    print(f"  最终种群大小: {len(results['population'])}")
    
    # 计算超体积
    objectives = results['population'].get_objective_matrix()
    hv = calculate_hypervolume(objectives)
    print(f"  超体积(HV): {hv:.6e}")
    
    # 打印种群信息
    print_population_info(results['population'], results['n_evaluations'])
    
    # 示例2: 使用自定义配置优化 ZDT2
    print("\n示例 2: 使用自定义配置优化 ZDT2")
    print("-" * 40)
    
    problem2 = ZDT2()
    
    # 自定义配置
    config = DiffusionEvolutionConfig(
        population_size=80,
        diffusion_steps=500,
        sample_size=30,
        hybrid_rate=0.4,
        training_epochs=5,
        condition_type='fitness'
    )
    
    algorithm2 = DiffusionEvolution(config)
    
    print("开始优化 ZDT2...")
    results2 = algorithm2.solve(problem2, max_evaluations=800, verbose=False)
    
    print(f"\n优化结果:")
    print(f"  评估次数: {results2['n_evaluations']}")
    print(f"  运行时间: {results2['runtime']:.2f}秒")
    
    objectives2 = results2['population'].get_objective_matrix()
    hv2 = calculate_hypervolume(objectives2)
    print(f"  超体积(HV): {hv2:.6e}")
    
    # 示例3: 多目标优化 DTLZ2
    print("\n示例 3: 多目标优化 DTLZ2")
    print("-" * 40)
    
    problem3 = DTLZ2(n_objectives=3, n_variables=12)
    
    # 多目标配置
    config3 = DiffusionEvolutionConfig(
        population_size=150,
        diffusion_steps=800,
        sample_size=60,
        hybrid_rate=0.4,
        condition_type='rank',
        noise_schedule='cosine'
    )
    
    algorithm3 = DiffusionEvolution(config3)
    
    print("开始优化 DTLZ2 (3目标)...")
    results3 = algorithm3.solve(problem3, max_evaluations=1200, verbose=False)
    
    print(f"\n优化结果:")
    print(f"  评估次数: {results3['n_evaluations']}")
    print(f"  运行时间: {results3['runtime']:.2f}秒")
    print(f"  最终种群大小: {len(results3['population'])}")
    
    # 示例4: 保存结果
    print("\n示例 4: 保存优化结果")
    print("-" * 40)
    
    save_results(results, 'zdt1_results.pkl')
    print("结果已保存到 zdt1_results.pkl")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_basic_example()