%% DiffusionEvolution for PlatEMO - MATLAB接口示例
% 这个文件展示了如何在PlatEMO中调用Python实现的DiffusionEvolution算法
%
% 作者: AI Assistant
% 日期: 2025
% 版本: 1.0.0

classdef DiffusionEvolution < ALGORITHM
    % <2025> <multi> <real> <constrained/none>
    % Diffusion Evolution - Python实现接口
    % 
    % 这是一个将扩散模型与进化算法相结合的多目标优化算法。
    % 该算法通过Python实现，通过MATLAB-Python接口调用。
    %
    %------------------------------- Parameters --------------------------------
    % N               - 种群大小 (默认: 100)
    % diffusion_steps - 扩散步数 (默认: 1000)
    % sample_size     - 每代采样数量 (默认: 50)
    % hybrid_rate     - 扩散解比例 (默认: 0.3)
    % model_type      - 扩散模型类型 ('DDPM' 或 'DDIM') (默认: 'DDPM')
    % training_epochs - 每代训练轮数 (默认: 10)
    % noise_schedule  - 噪声调度 ('linear' 或 'cosine') (默认: 'linear')
    % condition_type  - 条件类型 ('none', 'fitness', 'rank') (默认: 'fitness')
    %
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2025. Python实现版本。
    %--------------------------------------------------------------------------
    
    properties (Access = private)
        py_algorithm      % Python算法实例
        py_problem        % Python问题适配器
        config            % 配置参数
    end
    
    methods
        function main(Algorithm, Problem)
            %% 配置参数
            [N, diffusionSteps, sampleSize, hybridRate, modelType, trainingEpochs, noiseSchedule, conditionType] = ...
                Algorithm.ParameterSet(100, 1000, 50, 0.3, 'DDPM', 10, 'linear', 'fitness');
            
            % 保存配置
            Algorithm.config = struct(
                'population_size', N,
                'diffusion_steps', diffusionSteps,
                'sample_size', sampleSize,
                'hybrid_rate', hybridRate,
                'model_type', modelType,
                'training_epochs', trainingEpochs,
                'noise_schedule', noiseSchedule,
                'condition_type', conditionType
            );
            
            %% 初始化Python环境
            if ~exist('pyenv', 'file')
                error('需要MATLAB的Python接口支持。请先配置Python环境。');
            end
            
            % 确保Python模块在路径中
            module_path = fileparts(mfilename('fullpath'));
            python_path = fullfile(module_path, '..', '..', '..');
            
            if count(py.sys.path, python_path) == 0
                insert(py.sys.path, int32(0), python_path);
            end
            
            %% 初始化种群
            Population = Problem.Initialization();
            
            %% 创建Python算法实例
            config_json = jsonencode(Algorithm.config);
            Algorithm.py_algorithm = py.diffusion_evolution.matlab_adapter.MatlabDiffusionEvolution();
            
            % 配置算法
            result = Algorithm.py_algorithm.configure(config_json);
            result_struct = jsondecode(string(result));
            if strcmp(result_struct.status, 'error')
                error('算法配置失败: %s', result_struct.message);
            end
            
            %% 创建问题适配器
            problem_config = struct(
                'n_variables', Problem.D,
                'n_objectives', Problem.M,
                'lower_bound', Problem.lower',
                'upper_bound', Problem.upper',
                'has_constraints', Problem.constraint
            );
            
            problem_json = jsonencode(problem_config);
            result = Algorithm.py_algorithm.set_problem(problem_json);
            result_struct = jsondecode(string(result));
            if strcmp(result_struct.status, 'error')
                error('问题设置失败: %s', result_struct.message);
            end
            
            %% 设置评估回调
            Algorithm.py_algorithm.set_evaluate_callback(@(x) Algorithm.evaluate_python(x, Problem));
            
            %% 主优化循环
            while Algorithm.NotTerminated(Population)
                % 运行一代
                result_json = Algorithm.py_algorithm.solve(int32(Problem.N), 0, false);
                result = jsondecode(string(result_json));
                
                if strcmp(result.status, 'error')
                    warning('优化失败: %s', result.message);
                    break;
                end
                
                % 获取更新后的种群
                pop_json = Algorithm.py_algorithm.get_population();
                pop_data = jsondecode(string(pop_json));
                
                if strcmp(pop_data.status, 'success')
                    % 更新MATLAB种群
                    Population = Algorithm.update_matlab_population(Population, pop_data);
                end
            end
        end
    end
    
    methods (Access = private)
        function result = evaluate_python(obj, decision_vars, Problem)
            % Python评估回调函数
            
            % 转换为MATLAB数组
            if isa(decision_vars, 'py.numpy.ndarray')
                x = double(py.numpy.array(decision_vars));
            else
                x = double(decision_vars);
            end
            
            % 评估
            n_individuals = size(x, 1);
            objectives = zeros(n_individuals, Problem.M);
            constraints = [];
            
            for i = 1:n_individuals
                % 创建个体
                individual = Individual(x(i, :));
                
                % 评估
                individual = Problem.Evaluation(individual);
                
                % 提取结果
                objectives(i, :) = individual.objs;
                if Problem.constraint
                    constraints(i, :) = individual.cons;
                end
            end
            
            % 返回结果
            result = struct();
            result.objectives = objectives;
            if Problem.constraint
                result.constraints = constraints;
            end
        end
        
        function Population = update_matlab_population(obj, Population, pop_data)
            % 更新MATLAB种群
            
            % 提取数据
            decision_vars = pop_data.decision_vars;
            objectives = pop_data.objectives;
            constraints = pop_data.constraints;
            
            % 更新种群
            n_individuals = length(decision_vars);
            for i = 1:n_individuals
                if i <= length(Population)
                    % 更新现有个体
                    Population(i).decision = decision_vars{i};
                    Population(i).objs = objectives{i};
                    if ~isempty(constraints)
                        Population(i).cons = constraints{i};
                    end
                else
                    % 添加新个体
                    individual = Individual(decision_vars{i});
                    individual.objs = objectives{i};
                    if ~isempty(constraints)
                        individual.cons = constraints{i};
                    end
                    Population = [Population, individual];
                end
            end
        end
    end
end


%% 辅助函数 - 用于独立测试
function test_diffusion_evolution()
% 测试DiffusionEvolution算法

    fprintf('测试DiffusionEvolution算法\\n');
    fprintf('========================\\n\\n');
    
    % 检查Python环境
    if ~exist('pyenv', 'file')
        error('未找到Python接口。请先安装和配置Python。');
    end
    
    % 测试ZDT1问题
    fprintf('测试问题: ZDT1\\n');
    Problem = ZDT1();
    Algorithm = DiffusionEvolution('parameter', {50, 100, 20, 0.3, 'DDPM', 3, 'linear', 'fitness'});
    
    fprintf('开始优化...\\n');
    Algorithm.Solve(Problem);
    
    fprintf('\\n优化完成！\\n');
    fprintf('评估次数: %d\\n', Problem.FE);
    fprintf('最终种群大小: %d\\n', length(Algorithm.result{end}{2}));
    
    % 显示结果
    figure;
    Draw(Algorithm.result{end}{2}.objs, 'ro');
    title('DiffusionEvolution on ZDT1');
    xlabel('f_1');
    ylabel('f_2');
    grid on;
end


%% MATLAB评估函数示例
function result = evaluate_zdt1(decision_vars)
% ZDT1评估函数 - 供Python调用
    
    n_individuals = size(decision_vars, 1);
    objectives = zeros(n_individuals, 2);
    
    % 第一个目标
    objectives(:, 1) = decision_vars(:, 1);
    
    % 计算g函数
    g = 1 + 9 * mean(decision_vars(:, 2:end), 2);
    
    % 第二个目标
    objectives(:, 2) = g .* (1 - sqrt(decision_vars(:, 1) ./ g));
    
    result = struct();
    result.objectives = objectives;
end


%% 性能对比函数
function compare_algorithms()
% 对比DiffusionEvolution与其他算法

    fprintf('算法性能对比\\n');
    fprintf('============\\n\\n');
    
    % 问题列表
    problems = {'ZDT1', 'ZDT2', 'ZDT3'};
    algorithms = {@DiffusionEvolution, @NSGAII};
    
    % 运行对比
    for i = 1:length(problems)
        fprintf('问题: %s\\n', problems{i});
        Problem = eval(problems{i});
        
        for j = 1:length(algorithms)
            fprintf('  算法: %s\\n', func2str(algorithms{j}));
            
            Algorithm = algorithms{j}();
            if isa(Algorithm, 'DiffusionEvolution')
                Algorithm = DiffusionEvolution('parameter', {50, 100, 20, 0.3, 'DDPM', 3, 'linear', 'fitness'});
            end
            
            tic;
            Algorithm.Solve(Problem);
            runtime = toc;
            
            % 计算HV
            HV_value = HV(Algorithm.result{end}{2});
            
            fprintf('    HV: %.4e, Runtime: %.2fs\\n', HV_value, runtime);
        end
        fprintf('\\n');
    end
end