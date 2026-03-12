# DDD 3.0 - Dynamic Diffusion-Driven Multi-objective Optimization Algorithm

## 修复说明

本次修复解决了原代码中的语法错误，特别是第255行的无效表达式问题。

### 主要修复点：
1. 修复了 `trainNetwork` 方法中的矩阵拼接语法错误
2. 修复了 `sample` 方法中的输入拼接问题
3. 优化了代码结构，确保MATLAB语法正确

## 安装步骤

1. 将 `DDD3.0` 文件夹复制到 `PlatEMO/Algorithms/Multi-objective optimization/` 目录下
2. 在PlatEMO中选择DDD算法即可运行

## 文件说明

- `DDD.m` - 主算法文件
- `ConditionalDiffusionModel.m` - 条件扩散模型
- `SolutionArchive.m` - 精英解存档
- `AdaptiveScheduler.m` - 自适应调度器

## 使用说明

### 有Deep Learning Toolbox时
算法会正常使用扩散模型进行优化。

### 无Deep Learning Toolbox时
算法会自动切换到增强的GA-only模式，仍然可以正常运行并获得较好的结果。

## 参数设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| noise_schedule | [0.1, 0.01] | 噪声调度参数 |
| network_structure | [256, 512, 512, 256] | 神经网络结构 |
| ga_generations | 20 | GA初始采样代数 |
| sample_size | 500 | 训练样本大小 |
| dm_epochs | 100 | 扩散模型训练轮数 |
| dm_steps | 50 | 扩散采样步数 |
| archive_size | 1000 | 存档大小 |
| update_interval | 10 | 模型更新间隔 |
| dm_ratio | 0.4 | DM后代基础比例 |
| use_gpu | true | 是否使用GPU |

## 注意事项

1. 确保所有4个.m文件都在同一文件夹中
2. 算法需要MATLAB的统计工具箱（用于kmeans函数）
3. 扩散模型功能需要Deep Learning Toolbox（可选）
