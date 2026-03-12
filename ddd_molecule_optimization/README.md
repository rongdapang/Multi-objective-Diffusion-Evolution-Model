# DDD分子多目标优化

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](#english) | [中文](#中文)

---

## 中文

### 📖 简介

本项目实现了 **DDD (Diffusion-Driven Design)** 算法，一种结合**条件扩散模型**和**遗传算法**的多目标优化方法，用于分子设计。

**优化目标**:
- 🎯 **QED** (类药性) - 最大化
- 💧 **logP** (脂溶性) - 接近目标值
- ⚖️ **MW** (分子量) - 最小化

### 🚀 快速开始

#### 方式一：演示版本（无需RDKit）

```bash
python demo_main.py
```

#### 方式二：完整版本

```bash
# 安装依赖
pip install -r requirements.txt

# 运行优化
python main.py
```

### 📁 项目结构

```
ddd_molecule_optimization/
├── main.py                    # 主程序入口
├── demo_main.py               # 演示版本
├── data_utils.py              # 数据加载工具
├── molecule_encoder.py        # 分子编码器
├── diffusion_model.py         # 条件扩散模型
├── solution_archive.py        # 精英解存档
├── adaptive_scheduler.py      # 自适应调度器
├── evolution_optimizer.py     # 进化优化器
├── evaluation.py              # 评估和可视化
├── requirements.txt           # 依赖列表
├── install.sh                 # 安装脚本
├── 教学手册.md                # 详细教程
└── 项目报告.md                # 技术报告
```

### 📊 示例结果

运行后会生成以下文件：

```
results/
├── pareto_front.png           # 帕累托前沿图
├── convergence.png            # 收敛曲线
├── molecules.png              # 分子可视化
└── optimized_molecules.csv    # 优化结果
```

### 📚 文档

- [教学手册](教学手册.md) - 详细的使用教程
- [项目报告](项目报告.md) - 技术细节和算法说明

### 🔧 参数说明

```bash
python main.py \
    --data_path qm9.csv \        # 数据路径
    --num_samples 5000 \         # 样本数量
    --target_logp 2.0 \          # 目标logP值
    --pop_size 100 \             # 种群大小
    --n_gen 200 \                # 进化代数
    --dm_epochs 100 \            # 扩散模型训练轮数
    --dm_ratio 0.4 \             # 扩散模型比例
    --device cuda                # 计算设备
```

### 💻 系统要求

- Python 3.8+
- PyTorch 1.9+
- RDKit 2020.09+ (完整版本)
- 4GB+ RAM
- CUDA (可选)

---

## English

### 📖 Introduction

This project implements the **DDD (Diffusion-Driven Design)** algorithm, a multi-objective optimization method combining **conditional diffusion models** and **genetic algorithms** for molecular design.

**Optimization Objectives**:
- 🎯 **QED** (Drug-likeness) - Maximize
- 💧 **logP** (Lipophilicity) - Target value
- ⚖️ **MW** (Molecular Weight) - Minimize

### 🚀 Quick Start

#### Option 1: Demo Version (No RDKit required)

```bash
python demo_main.py
```

#### Option 2: Full Version

```bash
# Install dependencies
pip install -r requirements.txt

# Run optimization
python main.py
```

### 📁 Project Structure

```
ddd_molecule_optimization/
├── main.py                    # Main entry point
├── demo_main.py               # Demo version
├── data_utils.py              # Data loading utilities
├── molecule_encoder.py        # Molecular encoder
├── diffusion_model.py         # Conditional diffusion model
├── solution_archive.py        # Elite solution archive
├── adaptive_scheduler.py      # Adaptive scheduler
├── evolution_optimizer.py     # Evolutionary optimizer
├── evaluation.py              # Evaluation and visualization
├── requirements.txt           # Dependencies
├── install.sh                 # Installation script
├── 教学手册.md                # Detailed tutorial (Chinese)
└── 项目报告.md                # Technical report (Chinese)
```

### 📊 Example Results

After running, the following files will be generated:

```
results/
├── pareto_front.png           # Pareto front plot
├── convergence.png            # Convergence curve
├── molecules.png              # Molecular visualization
└── optimized_molecules.csv    # Optimization results
```

### 📚 Documentation

- [教学手册](教学手册.md) - Detailed tutorial (Chinese)
- [项目报告](项目报告.md) - Technical details (Chinese)

### 🔧 Parameters

```bash
python main.py \
    --data_path qm9.csv \        # Data path
    --num_samples 5000 \         # Number of samples
    --target_logp 2.0 \          # Target logP value
    --pop_size 100 \             # Population size
    --n_gen 200 \                # Number of generations
    --dm_epochs 100 \            # Diffusion model training epochs
    --dm_ratio 0.4 \             # Diffusion model ratio
    --device cuda                # Computing device
```

### 💻 System Requirements

- Python 3.8+
- PyTorch 1.9+
- RDKit 2020.09+ (full version)
- 4GB+ RAM
- CUDA (optional)

---

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- RDKit team for the cheminformatics toolkit
- Pymoo team for the multi-objective optimization framework

## 📧 Contact

For questions or suggestions, please open an issue or submit a pull request.

---

**Happy Optimizing!** 🎉
