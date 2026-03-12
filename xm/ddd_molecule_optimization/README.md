# DDD算法用于分子多目标优化

## 项目介绍

本项目实现了基于DDD（Diffusion-driven Design）算法的分子多目标优化系统，用于生成具有特定性质的分子。

## 环境与依赖

### 必需库
- Python 3.8+
- torch>=1.9.0
- rdkit>=2020.09.1
- numpy
- scipy
- matplotlib
- pandas
- tqdm
- scikit-learn
- pymoo>=0.6.0

### 可选库
- selfies（用于简化编码器）
- guacamol（用于预训练VAE模型）

## 安装依赖

```bash
pip install torch>=1.9.0 rdkit>=2020.09.1 numpy scipy matplotlib pandas tqdm scikit-learn pymoo>=0.6.0

# 可选
pip install selfies guacamol
```

## 数据准备

项目使用QM9数据集。如果本地没有QM9.csv文件，程序会自动生成示例数据用于演示。

## 运行命令

```bash
# 基本运行
python main.py

# 自定义参数运行
python main.py --num_samples 5000 --target_logp 2.0 --pop_size 100 --n_gen 200 --device cuda
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --data_path | str | qm9.csv | QM9数据路径 |
| --num_samples | int | 5000 | 使用样本数 |
| --target_logp | float | 2.0 | 目标logP值 |
| --pop_size | int | 100 | 种群大小 |
| --n_gen | int | 200 | 进化代数 |
| --dm_epochs | int | 100 | 扩散模型初始训练轮数 |
| --update_interval | int | 10 | 模型更新间隔 |
| --device | str | cuda | 设备类型（cuda/cpu） |

## 项目结构

- `data_utils.py`：数据加载和处理
- `molecule_encoder.py`：分子编码器-解码器
- `diffusion_model.py`：条件扩散模型
- `solution_archive.py`：精英存档
- `adaptive_scheduler.py`：自适应调度器
- `evolution_optimizer.py`：进化优化集成
- `evaluation.py`：评估与可视化
- `main.py`：主程序

## 输出结果

运行后会在`results`目录生成以下文件：
- `pareto_front.csv`：Pareto前沿分子的SMILES和性质
- `pareto_front.png`：Pareto前沿的可视化图

## 注意事项

1. 如果扩散模型训练失败，程序会自动切换到GA-only模式，仍可继续运行
2. 程序支持GPU加速，会自动检测设备类型
3. 为确保可复现性，程序固定了随机种子

## 预训练VAE模型

如果要使用预训练VAE模型，请将模型文件放置在项目根目录，并在`molecule_encoder.py`中修改模型加载路径。
