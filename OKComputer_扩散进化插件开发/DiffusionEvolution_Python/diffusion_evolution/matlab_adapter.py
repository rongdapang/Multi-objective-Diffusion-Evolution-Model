"""
MATLAB-Python 接口适配器

该模块提供了与 MATLAB/PlatEMO 平台交互的接口，
使得 Python 实现的 DiffusionEvolution 算法可以被 MATLAB 调用。
"""

import numpy as np
import json
import sys
from typing import Dict, Any, Optional, List
from .core import DiffusionEvolution, DiffusionEvolutionConfig, Population, Individual


class MatlabProblemAdapter:
    """
    MATLAB 问题适配器
    
    将 MATLAB 中的问题对象适配为 Python 格式
    """
    
    def __init__(self, problem_config: Dict[str, Any]):
        """
        初始化问题适配器
        
        参数:
            problem_config: 问题配置字典，包含:
                - n_variables: 决策变量数量
                - n_objectives: 目标函数数量
                - lower_bound: 下界数组
                - upper_bound: 上界数组
                - has_constraints: 是否有约束
                - constraint_types: 约束类型列表
        """
        self.n_variables = problem_config['n_variables']
        self.n_objectives = problem_config['n_objectives']
        self.lower_bound = np.array(problem_config['lower_bound'])
        self.upper_bound = np.array(problem_config['upper_bound'])
        self.has_constraints = problem_config.get('has_constraints', False)
        self.constraint_types = problem_config.get('constraint_types', [])
        
        # MATLAB 回调函数
        self.evaluate_callback = None
        
    def set_evaluate_callback(self, callback):
        """设置评估回调函数"""
        self.evaluate_callback = callback
        
    def evaluate(self, decision_vars: np.ndarray) -> Dict[str, np.ndarray]:
        """
        评估决策变量
        
        参数:
            decision_vars: 决策变量矩阵 (n_individuals, n_variables)
            
        返回:
            包含以下键的字典:
                - objectives: 目标函数值 (n_individuals, n_objectives)
                - constraints: 约束值 (n_individuals, n_constraints) 可选
        """
        if self.evaluate_callback is None:
            raise ValueError("未设置评估回调函数")
            
        # 调用 MATLAB 评估函数
        result = self.evaluate_callback(decision_vars)
        
        # 解析结果
        objectives = np.array(result['objectives'])
        
        output = {'objectives': objectives}
        
        if self.has_constraints and 'constraints' in result:
            output['constraints'] = np.array(result['constraints'])
            
        return output
        
    def initialize_population(self, n_individuals: int) -> np.ndarray:
        """
        初始化种群
        
        参数:
            n_individuals: 个体数量
            
        返回:
            决策变量矩阵 (n_individuals, n_variables)
        """
        return np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (n_individuals, self.n_variables)
        )


class MatlabDiffusionEvolution:
    """
    用于 MATLAB 调用的 DiffusionEvolution 包装类
    
    该类提供了与 MATLAB 兼容的接口，可以接收 MATLAB 的数据并返回结果。
    """
    
    def __init__(self):
        """初始化"""
        self.algorithm = None
        self.problem = None
        self.results = None
        
    def configure(self, config_json: str) -> str:
        """
        配置算法参数
        
        参数:
            config_json: JSON 格式的配置字符串
            
        返回:
            配置成功消息或错误信息
        """
        try:
            config_dict = json.loads(config_json)
            
            # 创建配置对象
            config = DiffusionEvolutionConfig(**config_dict)
            
            # 创建算法实例
            self.algorithm = DiffusionEvolution(config)
            
            return json.dumps({'status': 'success', 'message': '算法配置成功'})
            
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})
            
    def set_problem(self, problem_json: str) -> str:
        """
        设置优化问题
        
        参数:
            problem_json: JSON 格式的问题配置字符串
            
        返回:
            设置成功消息或错误信息
        """
        try:
            problem_config = json.loads(problem_json)
            self.problem = MatlabProblemAdapter(problem_config)
            
            return json.dumps({'status': 'success', 'message': '问题设置成功'})
            
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})
            
    def set_evaluate_callback(self, callback):
        """
        设置评估回调函数
        
        参数:
            callback: MATLAB 评估函数
        """
        if self.problem is not None:
            self.problem.set_evaluate_callback(callback)
            
    def solve(self, max_evaluations: int = 10000, max_runtime: float = float('inf'),
              verbose: bool = True) -> str:
        """
        运行优化
        
        参数:
            max_evaluations: 最大评估次数
            max_runtime: 最大运行时间（秒）
            verbose: 是否显示进度
            
        返回:
            JSON 格式的结果
        """
        if self.algorithm is None:
            return json.dumps({'status': 'error', 'message': '算法未配置'})
            
        if self.problem is None:
            return json.dumps({'status': 'error', 'message': '问题未设置'})
            
        try:
            # 运行优化
            results = self.algorithm.solve(
                self.problem,
                max_evaluations=max_evaluations,
                max_runtime=max_runtime,
                verbose=verbose
            )
            
            # 保存结果
            self.results = results
            
            # 转换为可序列化的格式
            result_data = {
                'status': 'success',
                'n_evaluations': int(results['n_evaluations']),
                'runtime': float(results['runtime']),
                'population_size': len(results['population']),
                'statistics': {
                    'training_loss': [float(x) for x in results['statistics']['training_loss']],
                    'population_diversity': [float(x) for x in results['statistics']['population_diversity']]
                }
            }
            
            return json.dumps(result_data)
            
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})
            
    def get_population(self) -> str:
        """
        获取最终种群
        
        返回:
            JSON 格式的种群数据
        """
        if self.results is None:
            return json.dumps({'status': 'error', 'message': '无可用结果'})
            
        try:
            population = self.results['population']
            
            # 提取数据
            decision_vars = population.get_decision_matrix().tolist()
            objectives = population.get_objective_matrix().tolist()
            
            # 约束信息
            constraints = None
            if population.get_constraint_matrix().size > 0:
                constraints = population.get_constraint_matrix().tolist()
                
            population_data = {
                'status': 'success',
                'decision_vars': decision_vars,
                'objectives': objectives,
                'constraints': constraints,
                'size': len(population)
            }
            
            return json.dumps(population_data)
            
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})
            
    def get_best_solutions(self, n_solutions: int = 10) -> str:
        """
        获取最优解
        
        参数:
            n_solutions: 要返回的解数量
            
        返回:
            JSON 格式的最优解数据
        """
        if self.results is None:
            return json.dumps({'status': 'error', 'message': '无可用结果'})
            
        try:
            population = self.results['population']
            
            # 选择最优解（第一前沿）
            selected_indices = self.algorithm.environmental_selection.select(population, 
                                                                             min(n_solutions, len(population)))
            
            # 提取数据
            decision_vars = population.get_decision_matrix()[selected_indices].tolist()
            objectives = population.get_objective_matrix()[selected_indices].tolist()
            
            best_data = {
                'status': 'success',
                'decision_vars': decision_vars,
                'objectives': objectives,
                'size': len(selected_indices)
            }
            
            return json.dumps(best_data)
            
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})
            
    def get_statistics(self) -> str:
        """
        获取统计信息
        
        返回:
            JSON 格式的统计信息
        """
        if self.results is None:
            return json.dumps({'status': 'error', 'message': '无可用结果'})
            
        try:
            stats = self.results['statistics']
            
            # 计算一些额外的统计信息
            population = self.results['population']
            objectives = population.get_objective_matrix()
            
            # 计算范围
            obj_ranges = []
            for m in range(objectives.shape[1]):
                obj_ranges.append({
                    'min': float(np.min(objectives[:, m])),
                    'max': float(np.max(objectives[:, m])),
                    'mean': float(np.mean(objectives[:, m]))
                })
                
            stats_data = {
                'status': 'success',
                'training_loss': stats['training_loss'],
                'population_diversity': stats['population_diversity'],
                'hybrid_rate_history': stats['hybrid_rate_history'],
                'objective_ranges': obj_ranges,
                'n_generations': len(stats['population_diversity'])
            }
            
            return json.dumps(stats_data)
            
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})


def run_from_matlab(config_json: str, problem_json: str, 
                    max_evaluations: int = 10000, max_runtime: float = float('inf')) -> str:
    """
    从 MATLAB 调用的主函数
    
    参数:
        config_json: 算法配置（JSON格式）
        problem_json: 问题配置（JSON格式）
        max_evaluations: 最大评估次数
        max_runtime: 最大运行时间
        
    返回:
        结果摘要（JSON格式）
    """
    try:
        # 创建算法实例
        algorithm = MatlabDiffusionEvolution()
        
        # 配置算法
        result = algorithm.configure(config_json)
        config_result = json.loads(result)
        if config_result['status'] != 'success':
            return result
            
        # 设置问题
        result = algorithm.set_problem(problem_json)
        problem_result = json.loads(result)
        if problem_result['status'] != 'success':
            return result
            
        # 运行优化
        result = algorithm.solve(max_evaluations=max_evaluations, max_runtime=max_runtime)
        
        return result
        
    except Exception as e:
        return json.dumps({'status': 'error', 'message': f'运行时错误: {str(e)}'})


# MATLAB 可以直接调用的函数
def diffusion_evolution_matlab(config_json: str, problem_json: str, 
                              evaluate_func, max_evaluations: int = 10000,
                              max_runtime: float = float('inf')) -> Dict[str, Any]:
    """
    MATLAB 调用接口
    
    参数:
        config_json: 算法配置（JSON格式）
        problem_json: 问题配置（JSON格式）
        evaluate_func: MATLAB 评估函数
        max_evaluations: 最大评估次数
        max_runtime: 最大运行时间
        
    返回:
        完整的优化结果
    """
    # 解析配置
    config_dict = json.loads(config_json)
    problem_config = json.loads(problem_json)
    
    # 创建问题适配器
    problem = MatlabProblemAdapter(problem_config)
    problem.set_evaluate_callback(evaluate_func)
    
    # 创建算法
    config = DiffusionEvolutionConfig(**config_dict)
    algorithm = DiffusionEvolution(config)
    
    # 运行优化
    results = algorithm.solve(problem, max_evaluations, max_runtime)
    
    return results