# 快速开始指南

## 🚀 项目简介

这是一个完整的扩散-进化模型实现，结合了扩散模型的生成能力和进化算法的优化能力。

## 📋 目录

- [环境准备](#环境准备)
- [快速测试](#快速测试)
- [基本使用](#基本使用)
- [高级功能](#高级功能)
- [常见问题](#常见问题)

## 🔧 环境准备

### 1. 安装依赖

```bash
# 安装Python依赖
pip install torch torchvision
pip install numpy matplotlib tqdm

# 可选：安装完整的依赖
pip install -r requirements.txt
```

### 2. 验证环境

```bash
# 运行快速测试
python experiments/run_experiment.py --mode quick_test
```

如果输出显示 "Quick test completed successfully!"，说明环境配置正确。

## ⚡ 快速测试

### 测试扩散模型

```bash
python experiments/run_experiment.py --mode quick_test
```

### 测试进化算法

```bash
python experiments/run_experiment.py --mode evolution_test
```

### 运行简单示例

```bash
# 快速演示（5轮训练）
python examples/simple_example.py --quick

# 完整示例（50轮训练）
python examples/simple_example.py
```

## 📖 基本使用

### 1. 创建模型

```python
from models.diffusion import DiffusionModel, UNet

# 创建UNet
unet = UNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    time_emb_dim=256
)

# 创建扩散模型
model = DiffusionModel(
    model=unet,
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
)
```

### 2. 训练模型

```python
from training.trainer import DiffusionEvolutionTrainer
from torch.utils.data import DataLoader

# 准备数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 创建训练器
trainer = DiffusionEvolutionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=2e-4,
    save_dir="./checkpoints",
    log_dir="./logs"
)

# 开始训练
history = trainer.train(num_epochs=200)
```

### 3. 生成样本

```python
# 生成样本
samples = trainer.generate_samples(num_samples=64)

# 保存样本
save_samples(samples, "./samples/generated.png", nrow=8)
```

### 4. 使用进化优化

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

## 🎯 高级功能

### 1. 使用配置文件

```python
from configs.config import ConfigManager, create_config_for_cifar10

# 创建配置
config_manager = create_config_for_cifar10()

# 修改配置
config_manager.training_config.num_epochs = 300
config_manager.model_config.base_channels = 128

# 保存配置
config_manager.save_config("./my_config.json")

# 加载配置
new_config = ConfigManager()
new_config.load_config("./my_config.json")
```

### 2. 自定义数据集

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path):
        # 加载数据
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 使用自定义数据集
dataset = MyDataset("./my_data")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 3. 自定义进化策略

```python
from models.evolution import SelectionStrategy

class MySelectionStrategy(SelectionStrategy):
    def select(self, population, num_parents):
        # 实现自定义选择策略
        pass

# 使用自定义选择策略
ga = GeneticAlgorithm(
    selection_strategy=MySelectionStrategy(),
    # ... 其他参数
)
```

### 4. 性能优化

```python
# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📊 监控和可视化

### 1. 训练曲线

```python
# 绘制训练曲线
plot_training_curves(
    train_losses=history['train_losses'],
    val_losses=history['val_losses'],
    save_path="./logs/training_curves.png"
)
```

### 2. 生成样本

```python
# 定期生成样本
if epoch % 50 == 0:
    trainer.generate_and_save_samples(epoch, num_samples=64)
```

### 3. 使用TensorBoard

```bash
# 安装TensorBoard
pip install tensorboard

# 启动TensorBoard
tensorboard --logdir=./logs
```

## 🔬 实验脚本

### 1. 训练CIFAR-10

```bash
python experiments/run_experiment.py --mode train --dataset cifar10 --epochs 200
```

### 2. 训练MNIST

```bash
python experiments/run_experiment.py --mode train --dataset mnist --epochs 100
```

### 3. 使用配置文件

```bash
python experiments/run_experiment.py --mode train --config ./my_config.json
```

## ❓ 常见问题

### Q: 训练速度慢怎么办？

**A:**
1. 使用GPU训练
2. 减小批次大小
3. 减少扩散步数
4. 使用更小的模型

```python
# 减少扩散步数
model = DiffusionModel(model=unet, timesteps=500)  # 默认1000

# 使用更小的模型
unet = UNet(base_channels=64)  # 默认128
```

### Q: 内存不足怎么办？

**A:**
1. 减小批次大小
2. 减小图像尺寸
3. 使用梯度累积
4. 启用检查点

```python
# 减小批次大小
train_loader = DataLoader(dataset, batch_size=16)  # 默认32

# 减小图像尺寸
transform = transforms.Resize((32, 32))
```

### Q: 如何提高生成质量？

**A:**
1. 增加训练轮数
2. 使用更大的模型
3. 调整学习率
4. 使用进化优化

```python
# 增加训练轮数
trainer.train(num_epochs=500)  # 默认200

# 使用更大的模型
unet = UNet(base_channels=256)  # 默认128

# 使用进化优化
evo_optimizer.optimize(num_generations=100)
```

### Q: 如何保存和加载模型？

**A:**

```python
# 保存模型
torch.save(model.state_dict(), "model.pth")

# 加载模型
model = DiffusionModel(model=unet)
model.load_state_dict(torch.load("model.pth"))

# 保存完整检查点
trainer.save_checkpoint(epoch=100, is_best=True)

# 加载检查点
trainer.load_checkpoint("./checkpoints/best_checkpoint.pth")
```

### Q: 如何使用自定义数据集？

**A:**

```python
from torchvision import transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
dataset = torchvision.datasets.ImageFolder(
    root="./my_dataset",
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 📚 进一步学习

### 1. 相关论文

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

### 2. 推荐阅读

- 扩散模型教程
- 进化算法导论
- 生成模型综述

### 3. 相关项目

- Stable Diffusion
- DALL-E 2
- Imagen

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

MIT License

## 📧 联系方式

如有问题或建议，请创建Issue或联系维护者。

## 🎉 开始使用

现在你已经了解了项目的基本使用方法，可以开始你的扩散-进化模型之旅了！

建议从以下步骤开始：

1. 运行快速测试验证环境
2. 运行简单示例了解基本流程
3. 尝试使用自己的数据集
4. 调整参数优化性能
5. 贡献代码改进项目

祝使用愉快！🚀