# DDD (Dynamic Diffusion-Driven) Algorithm - Fixed Version

基于PlatEMO平台的动态扩散驱动多目标优化算法实现（修复版）。

## 修复内容

### 1. Deep Learning Toolbox 检测问题修复

**原问题：**
- 使用 `license('test', 'Deep_Learning_Toolbox')` 检测可能不准确
- 检测到工具箱缺失后直接进入GA-only模式，没有增强的回退机制
- 缺少对所需函数存在性的检查

**修复方案：**
- 增加多重检测机制：
  - 许可证检查
  - 必需函数存在性检查 (`feedforwardnet`, `train`, `sim`, `mapminmax`)
  - 实际创建测试网络验证
- 添加增强的GA-only回退模式，包括：
  - 从初始采样中注入精英解
  - 引导变异增加多样性
  - 自适应参数调整

### 2. 错误处理增强

**新增功能：**
- 扩散模型训练失败时的优雅降级
- 采样过程中的错误捕获和处理
- 模型更新失败的容错机制
- 详细的警告信息帮助诊断问题

### 3. 代码结构优化

**改进：**
- 添加 `HasDLToolbox` 属性跟踪工具箱状态
- 分离检测逻辑到独立方法
- 改进代码可读性和维护性

## 算法概述

DDD算法结合了遗传算法(GA)和条件扩散模型(Diffusion Model)，通过在线模型适应实现协同优化。

### 主要特点

1. **四阶段流程**：GA初始采样 → 扩散模型训练 → 引导生成 → 混合进化
2. **动态适应**：扩散模型根据进化过程中发现的精英解持续更新
3. **自适应调度**：根据DM后代质量和优化停滞情况动态调整GA/DM贡献比例
4. **存档管理**：基于超体积贡献的精英解存档机制

## 文件结构

```
DDD2.0_fixed/
├── DDD.m                          # 主算法类（已修复）
├── SolutionArchive.m              # 精英解存档管理
├── ConditionalDiffusionModel.m    # 条件扩散模型（已修复）
├── AdaptiveScheduler.m            # 自适应调度器
└── README.md                      # 本文件
```

## 安装与使用

### 前置要求

- MATLAB R2018a 或更高版本
- PlatEMO 4.0 或更高版本
- Deep Learning Toolbox（可选，用于扩散模型训练）

### 安装步骤

1. 下载并安装 [PlatEMO](https://github.com/BIMK/PlatEMO)
2. 将 `DDD2.0_fixed` 文件夹复制到 `PlatEMO/Algorithms/Multi-objective optimization/` 目录下
3. 在PlatEMO中选择DDD算法即可运行

### 无Deep Learning Toolbox时的使用

当Deep Learning Toolbox不可用时，算法会自动切换到增强的GA-only模式：
- 使用高质量的初始GA采样
- 精英解存档机制仍然有效
- 自适应参数调整仍然工作
- 性能可能略低于完整模式，但仍具有竞争力

## 参数说明

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| noise_schedule | [0.1, 0.01] | 扩散模型噪声调度 [起始, 结束] |
| network_structure | [256, 512, 512, 256] | 扩散网络隐藏层结构 |
| ga_generations | 20 | 初始采样GA代数 |
| sample_size | 500 | 训练样本集合大小 |
| dm_epochs | 100 | 扩散模型训练轮数 |
| dm_steps | 50 | 扩散采样步数 |
| archive_size | 1000 | 精英解存档最大大小 |
| update_interval | 10 | 模型更新间隔代数 |
| dm_ratio | 0.4 | 扩散模型后代基础比例 |
| use_gpu | true | 是否使用GPU加速（如果可用） |

## 算法流程

```
1. 初始采样阶段
   └── 运行GA若干代收集高质量样本

2. 扩散模型训练阶段（如果DL Toolbox可用）
   └── 使用存档中的精英解训练条件扩散模型

3. 初始解生成阶段
   └── 使用扩散模型或GA生成初始种群

4. 主优化循环
   ├── 生成GA后代
   ├── 生成DM后代（如果可用）
   ├── 环境选择（NSGA-II机制）
   ├── 更新精英存档
   └── 按需更新扩散模型
```

## 故障排除

### 问题："Deep Learning Toolbox not found"

**解决方案：**
1. 安装Deep Learning Toolbox（推荐）
2. 或继续使用GA-only模式，算法会自动适应

### 问题："Diffusion model training failed"

**可能原因：**
- 训练数据不足
- 网络结构不适合问题

**解决方案：**
- 增加 `sample_size`
- 调整 `network_structure`
- 算法会自动回退到GA模式

### 问题：优化停滞

**解决方案：**
- 增加 `ga_generations` 改善初始采样
- 调整 `dm_ratio` 平衡GA和DM贡献
- 检查 `update_interval` 是否合适

## 引用

如果您在研究中使用了本算法，请引用：

```
[待添加论文引用]
```

## 版权

Copyright (c) 2026 BIMK Group.

本算法基于PlatEMO平台开发，使用请遵守PlatEMO的许可协议。

## 更新日志

### v2.0.1 (2026-03-08)
- 修复Deep Learning Toolbox检测问题
- 增强错误处理和回退机制
- 优化代码结构

### v2.0.0 (2026-03-08)
- 初始版本发布
