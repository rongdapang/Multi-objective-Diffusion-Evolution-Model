"""
MATLAB集成示例

展示如何在MATLAB中调用Python实现的DiffusionEvolution算法。
"""

import numpy as np
import json
from diffusion_evolution import DiffusionEvolution, DiffusionEvolutionConfig
from diffusion_evolution.matlab_adapter import MatlabDiffusionEvolution, MatlabProblemAdapter


def example_matlab_integration():
    """MATLAB集成示例"""
    print("MATLAB-PlatEMO集成示例")
    print("=" * 50)
    
    # 示例1: 使用适配器类
    print("\n示例1: 使用MatlabDiffusionEvolution适配器")
    print("-" * 40)
    
    # 配置算法
    config = {
        'population_size': 100,
        'diffusion_steps': 1000,
        'sample_size': 50,
        'hybrid_rate': 0.3,
        'model_type': 'DDPM',
        'training_epochs': 10,
        'noise_schedule': 'linear',
        'condition_type': 'fitness'
    }
    
    config_json = json.dumps(config)
    
    # 配置问题
    problem_config = {
        'n_variables': 30,
        'n_objectives': 2,
        'lower_bound': [0.0] * 30,
        'upper_bound': [1.0] * 30,
        'has_constraints': False
    }
    
    problem_json = json.dumps(problem_config)
    
    # 创建算法实例
    algorithm = MatlabDiffusionEvolution()
    
    # 配置算法
    result = algorithm.configure(config_json)
    print(f"配置结果: {result}")
    
    # 设置问题
    result = algorithm.set_problem(problem_json)
    print(f"问题设置结果: {result}")
    
    # 定义评估函数（模拟MATLAB回调）
    def evaluate_zdt1(decision_vars):
        """模拟ZDT1评估函数"""
        n_individuals = len(decision_vars)
        objectives = np.zeros((n_individuals, 2))
        
        # 第一个目标
        objectives[:, 0] = decision_vars[:, 0]
        
        # 计算g函数
        g = 1 + 9 * np.mean(decision_vars[:, 1:], axis=1)
        
        # 第二个目标
        objectives[:, 1] = g * (1 - np.sqrt(decision_vars[:, 0] / g))
        
        return {'objectives': objectives.tolist()}
    
    # 设置评估回调
    algorithm.set_evaluate_callback(evaluate_zdt1)
    
    # 运行优化
    print("\n开始优化...")
    result = algorithm.solve(max_evaluations=500, verbose=True)
    print(f"优化结果: {result}")
    
    # 获取结果
    population = algorithm.get_population()
    print(f"\n种群信息: {population[:200]}...")  # 截断显示
    
    # 获取最优解
    best_solutions = algorithm.get_best_solutions(n_solutions=5)
    print(f"\n最优解: {best_solutions[:200]}...")  # 截断显示
    
    # 获取统计信息
    statistics = algorithm.get_statistics()
    print(f"\n统计信息: {statistics[:200]}...")  # 截断显示


def example_matlab_json_interface():
    """MATLAB JSON接口示例"""
    print("\n示例2: MATLAB JSON接口")
    print("-" * 40)
    
    # 这个函数可以直接从MATLAB调用
    # MATLAB代码示例：
    #
    # config_json = '{"population_size": 100, ..., "condition_type": "fitness"}';
    # problem_json = '{"n_variables": 30, ..., "has_constraints": false}';
    # result = py.diffusion_evolution.matlab_adapter.run_from_matlab(...
    #     config_json, problem_json, @evaluate_function, int32(1000));
    #
    # results = jsondecode(string(result));
    
    print("MATLAB调用示例:")
    print("config_json = '{\"population_size\": 100, ...}';")
    print("problem_json = '{\"n_variables\": 30, ...}';")
    print("result = py.diffusion_evolution.matlab_adapter.run_from_matlab(...")
    print("    config_json, problem_json, @evaluate_function, int32(1000));")
    print("results = jsondecode(string(result));")


def example_matlab_callbacks():
    """MATLAB回调函数示例"""
    print("\n示例3: MATLAB回调函数示例")
    print("-" * 40)
    
    # 以下是MATLAB中需要定义的回调函数示例
    
    matlab_code = '''
    % MATLAB回调函数示例
    
    function result = evaluate_problem(decision_vars)
        % 评估函数
        % decision_vars: 决策变量矩阵 (n_individuals × n_variables)
        % result: 包含objectives和constraints的结构体
        
        n_individuals = size(decision_vars, 1);
        n_objectives = 2;
        
        objectives = zeros(n_individuals, n_objectives);
        
        % 目标函数计算
        objectives(:, 1) = decision_vars(:, 1);
        
        g = 1 + 9 * mean(decision_vars(:, 2:end), 2);
        objectives(:, 2) = g .* (1 - sqrt(decision_vars(:, 1) ./ g));
        
        result = struct();
        result.objectives = objectives;
        % result.constraints = ...; % 如果有约束
    end
    
    % 进度回调函数
    function progress_callback(generation, population, evaluations)
        fprintf('第%d代: 评估次数=%d\\n', generation, evaluations);
        % 可以在这里添加绘图代码
    end
    '''
    
    print("MATLAB回调函数定义:")
    print(matlab_code)


def example_data_transfer():
    """数据传输示例"""
    print("\n示例4: MATLAB-Python数据传输")
    print("-" * 40)
    
    # 模拟从MATLAB接收数据
    matlab_data = {
        'decision_vars': np.random.rand(100, 30).tolist(),
        'objectives': np.random.rand(100, 2).tolist(),
        'generation': 50,
        'evaluations': 5000
    }
    
    # 转换为JSON
    json_data = json.dumps(matlab_data)
    print(f"从MATLAB接收的数据大小: {len(json_data)} 字符")
    
    # 在Python中处理
    python_data = json.loads(json_data)
    decision_vars = np.array(python_data['decision_vars'])
    objectives = np.array(python_data['objectives'])
    
    print(f"决策变量形状: {decision_vars.shape}")
    print(f"目标函数形状: {objectives.shape}")
    print(f"代数: {python_data['generation']}")
    print(f"评估次数: {python_data['evaluations']}")


if __name__ == "__main__":
    print("DiffusionEvolution MATLAB集成示例")
    print("=" * 60)
    
    # 运行示例
    example_matlab_integration()
    example_matlab_json_interface()
    example_matlab_callbacks()
    example_data_transfer()
    
    print("\n" + "=" * 60)
    print("所有MATLAB集成示例完成！")
    print("=" * 60)