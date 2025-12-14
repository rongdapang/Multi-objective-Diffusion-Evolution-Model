# 使用指南

## 1. 快速开始（5分钟）

### 步骤1: 安装依赖

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### 步骤2: 运行快速演示

```bash
cd diffusion_evolution_model
python demo_quick.py
```

### 步骤3: 查看结果

运行后会看到：
- ✅ 扩散模型测试通过
- ✅ 进化算法测试通过
- ✅ 训练测试通过

## 2. 基本使用（10分钟）

### 2.1 创建和训练模型

```python
from models.diffusion import DiffusionModel, UNet
from training.trainer import DiffusionEvolutionTrainer
from torch.utils.data import DataLoader
import torch

# 创建设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型
unet = UNet(in_channels=3, out_channels=3, base_channels=64)
model = DiffusionModel(model=unet, timesteps=1000, device=device)

# 准备数据（使用随机数据作为示例）
data = torch.randn(100, 3, 32, 32)  # 100个32x32的彩色图像
train_loader = DataLoader(data, batch_size=16, shuffle=True)

# 创建训练器
trainer = DiffusionEvolutionTrainer(
    model=model,
    train_loader=train_loader,
    device=device,
    lr=2e-4
)

# 训练模型
history = trainer.train(num_epochs=50)

# 生成样本
samples = trainer.generate_samples(num_samples=16)
print(f"生成样本形状: {samples.shape}")
```

### 2.2 使用进化算法

```python
from models.evolution import GeneticAlgorithm, TournamentSelection

# 定义适应度函数
def fitness_function(genes):
    target = torch.ones_like(genes) * 0.5
    return -torch.sum((genes - target) ** 2).item()

# 创建遗传算法
ga = GeneticAlgorithm(
    population_size=30,
    gene_length=10,
    fitness_function=fitness_function,
    selection_strategy=TournamentSelection(),
    mutation_rate=0.1,
    crossover_rate=0.8
)

# 运行进化
best_individual = ga.run(num_generations=50)
print(f"最佳适应度: {best_individual.fitness}")
```

### 2.3 使用配置文件

```python
from configs.config import ConfigManager

# 创建配置管理器
config = ConfigManager()

# 修改配置
config.training_config.num_epochs = 100
config.model_config.base_channels = 128

# 保存配置
config.save_config("my_config.json")

# 加载配置
new_config = ConfigManager()
new_config.load_config("my_config.json")
```

## 3. 高级用法（30分钟）

### 3.1 自定义数据集

```python
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练
trainer = DiffusionEvolutionTrainer(model=model, train_loader=train_loader)
history = trainer.train(num_epochs=200)
```

### 3.2 使用进化优化

```python
from models.evolution import DiffusionEvolutionOptimizer

# 创建进化优化器
evo_optimizer = DiffusionEvolutionOptimizer(
    diffusion_model=model,
    population_size=20,
    mutation_rate=0.05,
    mutation_strength=0.01
)

# 运行优化
stats = evo_optimizer.optimize(num_generations=30)
print(f"优化统计: {stats}")
```

### 3.3 保存和加载模型

```python
# 保存模型
torch.save(model.state_dict(), "diffusion_model.pth")

# 加载模型
new_model = DiffusionModel(model=UNet(...))
new_model.load_state_dict(torch.load("diffusion_model.pth"))

# 保存完整检查点
trainer.save_checkpoint(epoch=100, is_best=True)

# 加载检查点
trainer.load_checkpoint("./checkpoints/best_checkpoint.pth")
```

## 4. 实验和演示

### 4.1 运行实验

```bash
# 训练CIFAR-10
python experiments/run_experiment.py --mode train --dataset cifar10 --epochs 200

# 进化算法测试
python experiments/run_experiment.py --mode evolution_test

# 快速测试
python experiments/run_experiment.py --mode quick_test
```

### 4.2 运行演示

```bash
# 快速演示
python demo_quick.py

# 完整演示
python demo.py

# 完整测试
python test_all.py
```

### 4.3 运行示例

```bash
# 简单示例
python examples/simple_example.py
```

## 5. 常见问题

### Q: 训练速度慢怎么办？

**A:**
```python
# 使用GPU
device = torch.device('cuda')
model.to(device)

# 减少扩散步数
model = DiffusionModel(model=unet, timesteps=500)

# 减小批次大小
train_loader = DataLoader(dataset, batch_size=16)

# 使用更小的模型
unet = UNet(base_channels=64)
```

### Q: 内存不足怎么办？

**A:**
```python
# 减小图像尺寸
transform = transforms.Resize((32, 32))

# 减小批次大小
train_loader = DataLoader(dataset, batch_size=8)

# 使用CPU
device = torch.device('cpu')
```

### Q: 如何提高生成质量？

**A:**
```python
# 增加训练轮数
trainer.train(num_epochs=500)

# 使用更大的模型
unet = UNet(base_channels=256)

# 调整学习率
trainer = DiffusionEvolutionTrainer(model=model, lr=1e-4)

# 使用进化优化
evo_optimizer.optimize(num_generations=100)
```

## 6. 可视化

### 6.1 生成样本

```python
from utils.utils import save_samples

# 生成并保存样本
samples = trainer.generate_samples(num_samples=64)
save_samples(samples, "samples.png", nrow=8)
```

### 6.2 训练曲线

```python
from utils.utils import plot_training_curves

# 绘制训练曲线
plot_training_curves(
    train_losses=history['train_losses'],
    val_losses=history.get('val_losses', []),
    save_path="training_curves.png"
)
```

### 6.3 使用TensorBoard

```bash
# 安装TensorBoard
pip install tensorboard

# 启动TensorBoard
tensorboard --logdir=./logs
```

## 7. 最佳实践

### 7.1 设置随机种子

```python
from utils.utils import set_seed

# 设置随机种子以确保可重复性
set_seed(42)
```

### 7.2 使用GPU

```python
# 自动检测和使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### 7.3 保存中间结果

```python
# 定期保存检查点
trainer.save_checkpoint(epoch=epoch, is_best=True)

# 定期生成样本
if epoch % 50 == 0:
    trainer.generate_and_save_samples(epoch)
```

### 7.4 监控训练

```python
# 使用日志记录
import logging
logger = logging.getLogger(__name__)
logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 8. 扩展开发

### 8.1 添加新模型

```python
from models.diffusion import DiffusionModel

class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 定义模型
    
    def forward(self, x, t):
        # 前向传播
        return output

# 使用新模型
model = DiffusionModel(model=MyModel(...))
```

### 8.2 添加新进化策略

```python
from models.evolution import SelectionStrategy

class MySelectionStrategy(SelectionStrategy):
    def select(self, population, num_parents):
        # 实现选择策略
        return parents

# 使用新策略
ga = GeneticAlgorithm(selection_strategy=MySelectionStrategy())
```

## 9. 性能调优

### 9.1 训练优化

```python
# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 9.2 内存优化

```python
# 梯度累积
accumulation_steps = 4
for i, data in enumerate(train_loader):
    loss = model(data) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 10. 部署

### 10.1 导出模型

```python
# 导出为ONNX
torch.onnx.export(model, input_tensor, "model.onnx")

# 导出为TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

### 10.2 创建API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    # 生成样本
    samples = model.generate(num_samples=1)
    return jsonify({'samples': samples.tolist()})

if __name__ == '__main__':
    app.run()
```

## 11. 故障排除

### 11.1 检查设备

```python
# 检查设备
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备数量: {torch.cuda.device_count()}")
print(f"当前设备: {torch.cuda.current_device()}")
```

### 11.2 检查模型

```python
# 检查模型参数
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数数量: {total_params:,}")

# 检查模型大小
model_size = total_params * 4 / 1024 / 1024
print(f"模型大小: {model_size:.2f} MB")
```

### 11.3 检查数据

```python
# 检查数据形状
print(f"数据形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
```

## 12. 获取更多帮助

### 12.1 查看文档
- **README.md**: 项目介绍
- **GETTING_STARTED.md**: 详细使用指南
- **ARCHITECTURE.md**: 系统架构

### 12.2 运行测试
```bash
python test_all.py
```

### 12.3 查看示例
```bash
python examples/simple_example.py
```

### 12.4 查看演示
```bash
python demo_quick.py
```

## 总结

这个项目提供了一个完整的扩散-进化模型实现，包括：

- ✅ 完整的扩散模型
- ✅ 多种进化算法
- ✅ 完整的训练系统
- ✅ 丰富的工具集
- ✅ 详细的文档
- ✅ 完整的测试

开始使用你的扩散-进化模型之旅吧！🚀