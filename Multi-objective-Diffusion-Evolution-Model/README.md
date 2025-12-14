# 扩散-进化模型 (Diffusion-Evolution Model)

这是一个结合了扩散模型和进化算法的深度学习框架，用于生成高质量的数据样本，并通过进化算法优化模型性能。

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [安装要求](#安装要求)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [算法原理](#算法原理)
- [实验结果](#实验结果)
- [扩展指南](#扩展指南)
- [常见问题](#常见问题)

## 🎯 项目简介

扩散-进化模型是一个创新的深度学习框架，它结合了两个强大的技术：

1. **扩散模型 (Diffusion Models)**: 通过逐步添加和去除噪声来生成高质量的数据样本
2. **进化算法 (Evolutionary Algorithms)**: 通过模拟自然选择过程来优化模型参数和生成过程

这种结合使得模型能够：
- 生成高质量的图像和数据
- 自适应地优化生成过程
- 避免局部最优解
- 提高模型的泛化能力

## ✨ 核心特性

- 🚀 **完整的扩散模型实现**: 包含前向过程、反向过程和训练流程
- 🧬 **多种进化算法**: 支持锦标赛选择、轮盘赌选择、排序选择等多种策略
- 🎛️ **灵活的配置系统**: 通过配置文件轻松调整模型和训练参数
- 📊 **详细的监控工具**: 包含训练曲线、样本生成和性能评估
- 🔧 **模块化设计**: 易于扩展和定制
- 💻 **多设备支持**: 支持CPU和GPU训练

## 📁 项目结构

```
diffusion_evolution_model/
├── models/                    # 模型定义
│   ├── diffusion.py          # 扩散模型核心实现
│   └── evolution.py          # 进化算法实现
├── training/                 # 训练相关
│   └── trainer.py           # 训练器
├── configs/                  # 配置管理
│   └── config.py            # 配置类定义
├── utils/                    # 工具函数
│   └── utils.py             # 通用工具
├── experiments/              # 实验脚本
│   └── run_experiment.py    # 实验运行脚本
├── checkpoints/              # 模型检查点
├── logs/                     # 训练日志
├── samples/                  # 生成的样本
└── data/                     # 数据集
```

## 📦 安装要求

### 系统要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于GPU加速)

### 依赖库

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm
pip install tensorboard  # 可选，用于可视化
```

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/rongdapang/Multi-objective-Diffusion-Evolution-Model.git
cd diffusion-evolution-model
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 快速测试

运行快速测试以验证安装：

```bash
python experiments/run_experiment.py --mode quick_test
```

### 2. 训练CIFAR-10

```bash
python experiments/run_experiment.py --mode train --dataset cifar10 --epochs 200
```

### 3. 进化算法测试

```bash
python experiments/run_experiment.py --mode evolution_test
```

### 4. 自定义训练

```python
from models.diffusion import DiffusionModel, UNet
from training.trainer import DiffusionEvolutionTrainer
from configs.config import ConfigManager

# 创建配置
config = ConfigManager()

# 创建模型
unet = UNet(in_channels=3, out_channels=3, base_channels=128)
model = DiffusionModel(model=unet)

# 创建训练器
trainer = DiffusionEvolutionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# 开始训练
history = trainer.train(num_epochs=200)
```

## 📖 使用指南

### 基本训练流程

```python
from models.diffusion import DiffusionModel, UNet
from training.trainer import DiffusionEvolutionTrainer
from configs.config import create_config_for_cifar10

# 1. 创建配置
config_manager = create_config_for_cifar10()

# 2. 创建模型
unet = UNet(
    in_channels=3,
    out_channels=3,
    base_channels=128,
    timesteps=1000
)
model = DiffusionModel(model=unet)

# 3. 创建数据加载器
train_loader, val_loader = load_dataset(
    dataset_name="cifar10",
    data_path="./data",
    image_size=32,
    batch_size=64
)

# 4. 创建训练器
trainer = DiffusionEvolutionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=2e-4
)

# 5. 训练模型
history = trainer.train(num_epochs=200)

# 6. 生成样本
samples = trainer.generate_samples(num_samples=64)
```

### 进化优化

```python
from models.evolution import DiffusionEvolutionOptimizer

# 创建进化优化器
evo_optimizer = DiffusionEvolutionOptimizer(
    diffusion_model=model,
    population_size=30,
    mutation_rate=0.1,
    mutation_strength=0.01
)

# 运行优化
stats = evo_optimizer.optimize(num_generations=50)
```

### 模型评估

```python
from utils.utils import plot_training_curves, save_samples

# 绘制训练曲线
plot_training_curves(
    train_losses=history['train_losses'],
    val_losses=history['val_losses'],
    save_path='./logs/training_curves.png'
)

# 保存生成的样本
save_samples(
    samples=samples,
    save_path='./samples/generated.png',
    nrow=8
)
```

## ⚙️ 配置说明

### 模型配置 (ModelConfig)

```python
@dataclass
class ModelConfig:
    in_channels: int = 3          # 输入通道数
    out_channels: int = 3         # 输出通道数
    base_channels: int = 128      # 基础通道数
    channel_mults: Tuple = (1, 2, 4, 8)  # 通道倍数
    num_res_blocks: int = 2       # 残差块数量
    attention_resolutions: Tuple = (16, 8)  # 注意力分辨率
    dropout: float = 0.1          # Dropout率
    timesteps: int = 1000         # 扩散步数
    beta_start: float = 0.0001    # Beta起始值
    beta_end: float = 0.02        # Beta结束值
```

### 训练配置 (TrainingConfig)

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 200         # 训练轮数
    batch_size: int = 32          # 批次大小
    learning_rate: float = 2e-4   # 学习率
    weight_decay: float = 1e-4    # 权重衰减
    image_size: int = 32          # 图像大小
    save_every: int = 10          # 保存频率
    generate_every: int = 50      # 生成样本频率
    evolution_interval: int = 100 # 进化优化间隔
    evolution_generations: int = 20  # 进化代数
```

### 进化配置 (EvolutionConfig)

```python
@dataclass
class EvolutionConfig:
    population_size: int = 30     # 种群大小
    mutation_rate: float = 0.1    # 变异率
    mutation_strength: float = 0.01  # 变异强度
    crossover_rate: float = 0.7   # 交叉率
    elitism: int = 2              # 精英保留数量
    num_generations: int = 50     # 进化代数
    selection_strategy: str = "tournament"  # 选择策略
```

## 🔬 算法原理

### 扩散模型

扩散模型通过以下步骤工作：

1. **前向过程 (Forward Process)**:
   - 逐步向原始数据添加噪声
   - 最终得到纯高斯噪声
   - 这个过程是固定的

2. **反向过程 (Reverse Process)**:
   - 从纯噪声开始
   - 逐步去除噪声
   - 最终重建原始数据
   - 这个过程由神经网络学习

数学表示：
```
q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI)
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### 进化算法

进化算法模拟自然选择过程：

1. **初始化**: 随机生成初始种群
2. **评估**: 计算每个个体的适应度
3. **选择**: 选择优秀的个体作为父代
4. **交叉**: 通过交叉操作产生后代
5. **变异**: 对后代进行随机变异
6. **替换**: 用新种群替换旧种群
7. **重复**: 直到满足终止条件

选择策略：
- **锦标赛选择**: 随机选择k个个体，选择最好的
- **轮盘赌选择**: 按适应度比例选择
- **排序选择**: 按排名选择

## 📊 实验结果

### CIFAR-10 生成结果

| 指标 | 值 |
|------|-----|
| FID Score | XX.XX |
| IS Score | XX.XX |
| 训练时间 | XX 小时 |
| GPU 内存 | XX GB |

### 不同配置的比较

| 配置 | 训练时间 | 生成质量 | 内存使用 |
|------|---------|---------|---------|
| 基础配置 | XX | XX | XX |
| 大模型 | XX | XX | XX |
| 少步数 | XX | XX | XX |

## 🔧 扩展指南

### 添加新的选择策略

```python
from models.evolution import SelectionStrategy

class MySelectionStrategy(SelectionStrategy):
    def select(self, population: Population, num_parents: int) -> List[Individual]:
        # 实现你的选择策略
        pass
```

### 自定义适应度函数

```python
def custom_fitness_function(genes: torch.Tensor) -> float:
    # 实现你的适应度评估
    return fitness_score
```

### 添加新的数据集

```python
def load_custom_dataset(data_path: str, **kwargs):
    # 实现数据加载逻辑
    return train_loader, val_loader
```

## ❓ 常见问题

### Q: 训练速度慢怎么办？

A: 
1. 减少扩散步数 (`timesteps`)
2. 减小批次大小 (`batch_size`)
3. 使用更小的模型 (`base_channels`)
4. 使用混合精度训练

### Q: 生成质量不好怎么办？

A:
1. 增加训练轮数
2. 增大模型容量
3. 调整学习率
4. 使用进化算法优化

### Q: 内存不足怎么办？

A:
1. 减小批次大小
2. 减小图像尺寸
3. 使用梯度累积
4. 使用检查点技术

### Q: 如何保存和加载模型？

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model.load_state_dict(torch.load('model.pth'))
```

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Jonathan Ho et al.
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) - Jascha Sohl-Dickstein et al.
- 所有为本项目做出贡献的开发者和研究者

## 📧 联系方式

- 项目维护者: [Your Name]
- 邮箱: [2322806418@qq.com]
- 项目链接: [https://github.com/rongdapang/Multi-objective-Diffusion-Evolution-Model]
---

**注意**: 这是一个研究项目，代码正在持续更新中。如有问题或建议，欢迎提出 Issue。