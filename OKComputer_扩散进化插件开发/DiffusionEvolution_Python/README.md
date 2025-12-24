# DiffusionEvolution - Python版本

## 项目简介

**DiffusionEvolution** 是一个创新的多目标优化算法，将扩散模型的生成能力与传统的进化算法相结合。该算法专为 PlatEMO 平台设计，支持通过 MATLAB-Python 接口进行调用。

### 核心创新

1. **扩散模型作为智能变异算子**：学习种群分布特征，生成高质量的后代
2. **混合进化策略**：平衡探索与开发，结合传统遗传操作和扩散生成
3. **条件生成机制**：基于适应度或排序信息指导生成过程
4. **自适应机制**：根据优化进度动态调整参数

### 算法架构

```
┌─────────────────────────────────────────────────────────┐
│              DiffusionEvolution Algorithm               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │初始化       │  │训练扩散     │  │生成混合     │   │
│  │种群和模型   │─→│模型         │─→│后代         │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
│         ↓                                  ↓          │
│  ┌─────────────┐                  ┌─────────────┐     │
│  │环境选择     │◀─────────────────│更新训练     │     │
│  │(非支配排序) │                  │数据         │     │
│  └─────────────┘                  └─────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 安装指南

### 系统要求

- Python 3.7 或更高版本
- NumPy 1.19.0 或更高版本
- MATLAB R2018b 或更高版本（用于PlatEMO集成）
- PlatEMO 4.0 或更高版本

### 安装步骤

#### 1. Python环境准备

```bash
# 创建虚拟环境
python -m venv diffusion_env
source diffusion_env/bin/activate  # Linux/Mac
# 或
diffusion_env\Scripts\activate     # Windows

# 安装依赖
pip install numpy
pip install scipy
```

#### 2. 安装DiffusionEvolution

```bash
# 下载代码
cd DiffusionEvolution_Python

# 安装包
pip install -e .
```

#### 3. MATLAB配置

在 MATLAB 中设置 Python 环境：

```matlab
% 设置Python路径
pyenv('Version', 'path/to/your/python.exe');

% 添加Python模块路径
if count(py.sys.path,'path/to/DiffusionEvolution_Python') == 0
    insert(py.sys.path,int32(0),'path/to/DiffusionEvolution_Python');
end
```

## 快速开始

### 5分钟快速入门

#### 1. 独立Python使用

```python
from diffusion_evolution import DiffusionEvolution, DiffusionEvolutionConfig
import numpy as np

# 定义简单测试问题
class TestProblem:
    def __init__(self):
        self.n_variables = 2
        self.n_objectives = 2
        self.lower_bound = np.array([0, 0])
        self.upper_bound = np.array([1, 1])
    
    def evaluate(self, decision_vars):
        # ZDT1 测试问题
        n = len(decision_vars)
        objectives = np.zeros((n, 2))
        
        objectives[:, 0] = decision_vars[:, 0]
        g = 1 + 9 * np.mean(decision_vars[:, 1:], axis=1)
        objectives[:, 1] = g * (1 - np.sqrt(decision_vars[:, 0] / g))
        
        return {'objectives': objectives}

# 创建问题
problem = TestProblem()

# 使用默认配置
algorithm = DiffusionEvolution()

# 运行优化
results = algorithm.solve(problem, max_evaluations=1000)

# 显示结果
print(f"评估次数: {results['n_evaluations']}")
print(f"运行时间: {results['runtime']:.2f}秒")
print(f"最终种群大小: {len(results['population'])}")
```

#### 2. MATLAB-PlatEMO集成使用

在 MATLAB 中创建 Python 接口：

```matlab
% 配置算法参数
config = struct();
config.population_size = 100;
config.diffusion_steps = 1000;
config.sample_size = 50;
config.hybrid_rate = 0.3;
config.model_type = 'DDPM';
config.training_epochs = 10;
config.noise_schedule = 'linear';
config.condition_type = 'fitness';

config_json = jsonencode(config);

% 配置问题
problem_config = struct();
problem_config.n_variables = 30;
problem_config.n_objectives = 2;
problem_config.lower_bound = zeros(1, 30);
problem_config.upper_bound = ones(1, 30);
problem_config.has_constraints = false;

problem_json = jsonencode(problem_config);

% 创建评估函数
evaluate_func = @(x) py.diffusion_evolution.matlab_adapter.evaluate_zdt1(x);

% 运行优化
result_json = py.diffusion_evolution.matlab_adapter.diffusion_evolution_matlab(...
    config_json, problem_json, evaluate_func, int32(10000));

% 解析结果
results = jsondecode(string(result_json));
```

### 完整示例

查看 `examples/` 目录获取完整的使用示例：

- `example_basic.py` - 基础使用示例
- `example_advanced.py` - 高级配置示例
- `example_matlab.py` - MATLAB集成示例
- `example_comparison.py` - 算法对比示例

## 算法参数详解

### 核心参数

| 参数名 | 说明 | 默认值 | 推荐范围 |
|--------|------|--------|----------|
| `population_size` | 种群大小 | 100 | 50-300 |
| `diffusion_steps` | 扩散步数 | 1000 | 500-2000 |
| `sample_size` | 每代采样数量 | 50 | 20-100 |
| `hybrid_rate` | 扩散解比例 | 0.3 | 0.2-0.6 |
| `model_type` | 扩散模型类型 | 'DDPM' | 'DDPM', 'DDIM' |
| `training_epochs` | 训练轮数 | 10 | 5-20 |
| `noise_schedule` | 噪声调度 | 'linear' | 'linear', 'cosine' |
| `condition_type` | 条件类型 | 'fitness' | 'none', 'fitness', 'rank' |

### 高级参数

| 参数名 | 说明 | 默认值 | 推荐范围 |
|--------|------|--------|----------|
| `adaptive_diffusion` | 自适应扩散 | True | True/False |
| `memory_size` | 训练数据容量 | 1000 | 500-2000 |
| `diffusion_strength` | 扩散强度 | 0.1 | 0.05-0.2 |
| `min_hybrid_rate` | 最小混合比例 | 0.1 | 0.0-0.3 |
| `max_hybrid_rate` | 最大混合比例 | 0.5 | 0.4-0.8 |

### 参数配置建议

#### 1. 双目标问题

```python
config = DiffusionEvolutionConfig(
    population_size=100,
    diffusion_steps=1000,
    hybrid_rate=0.3,
    condition_type='fitness',
    noise_schedule='linear'
)
```

#### 2. 多目标问题（3+目标）

```python
config = DiffusionEvolutionConfig(
    population_size=200,
    diffusion_steps=1500,
    hybrid_rate=0.4,
    condition_type='rank',
    noise_schedule='cosine'
)
```

#### 3. 约束优化问题

```python
config = DiffusionEvolutionConfig(
    population_size=150,
    diffusion_steps=1000,
    hybrid_rate=0.3,
    condition_type='fitness',
    adaptive_diffusion=True
)
```

#### 4. 快速测试配置

```python
config = DiffusionEvolutionConfig(
    population_size=50,
    diffusion_steps=100,
    hybrid_rate=0.3,
    training_epochs=3,
    condition_type='none'
)
```

## 算法原理

### 扩散模型基础

扩散模型通过以下过程学习数据分布：

1. **前向过程**：逐步向数据添加噪声，直到变成纯噪声
   ```
   q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)
   ```

2. **反向过程**：学习如何从噪声恢复原始数据
   ```
   p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
   ```

### 在优化中的应用

#### 1. 训练阶段
- 使用当前种群作为训练数据
- 学习种群的分布特征
- 可选：使用适应度或排序信息作为条件

#### 2. 生成阶段
- 从噪声开始反向扩散
- 生成符合学习分布的新解
- 保证解的质量和多样性

#### 3. 混合策略
- 扩散生成：定向搜索，利用学习信息
- 遗传操作：全局搜索，保持探索能力
- 自适应调整：根据优化进度动态平衡

## 优化方向建议

### 1. 性能优化

#### 计算效率提升
```python
# 使用更少的扩散步数
config.diffusion_steps = 500

# 减少训练轮数
config.training_epochs = 5

# 使用DDIM加速采样
config.model_type = 'DDIM'
```

#### 内存优化
```python
# 限制训练数据大小
config.memory_size = 500

# 使用较小的种群
config.population_size = 50
```

### 2. 算法改进

#### 扩散模型架构优化
- 使用更复杂的神经网络架构
- 添加注意力机制
- 改进条件嵌入方式

#### 自适应机制增强
- 动态调整混合比例
- 自适应噪声调度
- 基于多样性的参数调整

#### 约束处理改进
- 专门的条件生成策略
- 修复不可行解的机制
- 多阶段优化策略

### 3. 集成优化

#### PlatEMO集成优化
- 更好的MATLAB-Python数据传输
- 并行评估支持
- GUI界面集成

#### 多算法融合
- 与MOEA/D结合
- 与NSGA-III结合
- 混合策略优化

## 应用方向建议

### 1. 工程优化领域

#### 结构设计优化
- **应用场景**：桥梁、建筑、机械结构
- **优化目标**：重量、成本、安全性
- **预期效果**：找到更优的设计方案

#### 参数调优
- **应用场景**：控制器参数、工艺参数
- **优化目标**：性能指标、稳定性、鲁棒性
- **预期效果**：自动化参数优化

### 2. 机器学习领域

#### 超参数优化
- **应用场景**：深度学习模型调优
- **优化目标**：准确率、训练时间、模型复杂度
- **预期效果**：提高模型性能

#### 特征选择
- **应用场景**：高维数据降维
- **优化目标**：分类准确率、特征数量
- **预期效果**：选择最优特征子集

### 3. 金融优化领域

#### 投资组合优化
- **应用场景**：资产配置
- **优化目标**：收益、风险、流动性
- **预期效果**：构建最优投资组合

#### 风险管理
- **应用场景**：风险控制策略
- **优化目标**：风险最小化、收益最大化
- **预期效果**：平衡风险与收益

### 4. 能源优化领域

#### 能源调度
- **应用场景**：智能电网、可再生能源
- **优化目标**：成本、效率、环境影响
- **预期效果**：优化能源分配

#### 建筑节能
- **应用场景**：建筑设计、设备选型
- **优化目标**：能耗、舒适度、成本
- **预期效果**：绿色建筑优化

### 5. 生物医药领域

#### 药物设计
- **应用场景**：分子结构设计
- **优化目标**：药效、副作用、成本
- **预期效果**：加速药物发现

#### 治疗方案优化
- **应用场景**：个性化医疗
- **优化目标**：治疗效果、副作用、费用
- **预期效果**：个性化治疗方案

## 常见问题

### Q1: 算法运行很慢怎么办？

**A1**: 尝试以下优化：
```python
# 减少扩散步数
config.diffusion_steps = 500

# 减少训练轮数
config.training_epochs = 5

# 减少种群大小
config.population_size = 50
```

### Q2: 收敛效果不好怎么办？

**A2**: 尝试以下调整：
```python
# 增加混合比例
config.hybrid_rate = 0.5

# 使用余弦调度
config.noise_schedule = 'cosine'

# 增加训练轮数
config.training_epochs = 15
```

### Q3: 内存不足怎么办？

**A3**: 限制训练数据大小：
```python
# 减少内存使用
config.memory_size = 500

# 减少种群大小
config.population_size = 50
```

### Q4: 如何处理约束问题？

**A4**: 使用条件生成：
```python
# 使用适应度条件
config.condition_type = 'fitness'

# 启用自适应
config.adaptive_diffusion = True
```

## 版本信息

- **版本**: 1.0.0
- **发布日期**: 2025年
- **支持平台**: Python 3.7+, MATLAB R2018b+, PlatEMO 4.0+
- **许可证**: MIT License

## 引用

如果您在研究中使用本算法，请引用：

```bibtex
@software{diffusionevolution2025,
  title={DiffusionEvolution: 扩散模型与进化算法混合的多目标优化},
  author={AI Assistant},
  year={2025},
  version={1.0.0},
  url={https://github.com/your-repo/DiffusionEvolution_Python}
}
```

## 联系与支持

- **问题反馈**: [GitHub Issues](https://github.com/your-repo/issues)
- **文档**: [完整文档](docs/)
- **示例**: [examples/](examples/)

## 更新日志

### v1.0.0 (2025)
- 初始版本发布
- 支持扩散模型集成
- 支持MATLAB-PlatEMO接口
- 完整文档和示例

---

**祝您使用愉快！如有问题，欢迎反馈。**