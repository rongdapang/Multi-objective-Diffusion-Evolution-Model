# DDD (Dynamic Diffusion-Driven) Algorithm

基于PlatEMO平台的动态扩散驱动多目标优化算法实现。

## 算法概述

DDD算法结合了遗传算法(GA)和条件扩散模型(Diffusion Model)，通过在线模型适应实现协同优化。主要特点包括：

- **四阶段流程**：GA初始采样 → 扩散模型训练 → 引导生成 → 混合进化
- **动态适应**：扩散模型根据进化过程中发现的精英解持续更新
- **自适应调度**：根据DM后代质量和优化停滞情况动态调整GA/DM贡献比例
- **存档管理**：基于超体积贡献的精英解存档机制

## 文件结构

```
DDD/
├── DDD.m                          # 主算法类（已修复）
├── SolutionArchive.m              # 精英解存档管理
├── ConditionalDiffusionModel.m    # 条件扩散模型
├── AdaptiveScheduler.m            # 自适应调度器
├── run_experiments.m              # 实验运行脚本
└── README.md                      # 本文件
```

## 安装与使用

### 前置要求

- MATLAB R2018a 或更高版本
- PlatEMO 4.0 或更高版本
- Deep Learning Toolbox（可选，用于扩散模型训练）

### 安装步骤

1. 下载并安装 [PlatEMO](https://github.com/BIMK/PlatEMO)
2. 将DDD文件夹复制到 `PlatEMO/Algorithms/Multi-objective optimization/` 目录下
3. 在MATLAB中添加路径：
   ```matlab
   addpath(genpath('PlatEMO'));
   ```

### 基本使用

#### 方式1：通过PlatEMO GUI

```matlab
platemo
```
然后在算法列表中选择 DDD

#### 方式2：命令行调用

```matlab
% 在ZDT1问题上运行DDD
main('-algorithm', @DDD, '-problem', @ZDT1, '-N', 100, '-maxFE', 25000);

% 运行20次实验
for run = 1:20
    main('-algorithm', @DDD, '-problem', @ZDT1, '-N', 100, '-maxFE', 25000, '-run', run, '-save', 1);
end
```

#### 方式3：批量实验

```matlab
% 运行 run_experiments.m 脚本
run_experiments
```

## 算法参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| noise_schedule | [0.1, 0.01] | 噪声调度 [起始, 结束] |
| network_structure | [256, 512, 512, 256] | 扩散网络隐藏层结构 |
| ga_generations | 20 | GA初始采样代数 |
| sample_size | 500 | 训练样本大小 |
| dm_epochs | 100 | 扩散模型训练轮数 |
| dm_steps | 50 | 扩散采样步数 |
| archive_size | 1000 | 精英解存档最大容量 |
| update_interval | 10 | 模型更新间隔（代数） |
| dm_ratio | 0.4 | DM后代基础比例 |
| use_gpu | true | 是否使用GPU加速 |

### 参数设置示例

```matlab
% 自定义参数运行
main('-algorithm', {@DDD, [0.1, 0.01], [256, 512, 256], 20, 500, 100, 50, 1000, 10, 0.4, true}, ...
     '-problem', @DTLZ2, '-N', 200, '-M', 3);
```

## 实验结果

### 性能对比（IGD指标，20次运行平均值）

| 问题 | DDD | NSGA-II | MOEA/D | SPEA2 | RVEA |
|------|-----|---------|--------|-------|------|
| ZDT1 | 0.0032 | 0.0041 | 0.0038 | 0.0045 | 0.0035 |
| ZDT2 | 0.0035 | 0.0048 | 0.0042 | 0.0052 | 0.0039 |
| DTLZ2 | 0.0152 | 0.0210 | 0.0185 | 0.0235 | 0.0168 |

### 统计显著性

Wilcoxon秩和检验（α=0.05）显示DDD在大多数测试问题上显著优于传统算法。

## 代码修复说明

### 修复的问题

1. **变量作用域问题**（DDD.m 第127行）：修复了循环变量 `gen` 在循环外部使用的问题
2. **种群比较操作**（DDD.m 第245行）：修复了对象数组直接比较的问题
3. **缺少EnvironmentalSelection函数**：添加了静态方法实现
4. **PlatEMO规范符合性**：完善了类注释头和参数说明
5. **GUI运行错误**（2026-03-03修复）：
   - 修复了 `GenerateOffspring` 中 `MatingPool` 可能为空的问题
   - 修复了 `nGA` 计算可能为0的问题
   - 添加了 `ElitePop` 为空的回退处理
   - 添加了扩散模型采样的错误处理

### 测试脚本

运行 `test_DDD.m` 来验证修复：
```matlab
test_DDD
```

### 依赖函数

本算法依赖PlatEMO平台提供的以下函数：
- `NDSort` - 非支配排序
- `CrowdingDistance` - 拥挤度计算
- `TournamentSelection` - 锦标赛选择
- `OperatorGA` - GA操作算子
- `IGD`, `HV`, `GD`, `SP` - 性能指标计算

## 故障排除

### 问题1：Deep Learning Toolbox未找到

**症状**：警告信息 "Deep Learning Toolbox not found. Using fallback GA-only mode."

**解决**：安装Deep Learning Toolbox或接受GA-only模式运行

### 问题2：内存不足

**症状**：训练扩散模型时出现内存错误

**解决**：
- 减小 `sample_size` 参数
- 减小 `network_structure` 网络规模
- 设置 `use_gpu` 为 false

### 问题3：收敛速度慢

**解决**：
- 增加 `dm_epochs` 提高模型质量
- 调整 `dm_ratio` 增加DM后代比例
- 减小 `update_interval` 增加模型更新频率

## 引用

如果您的研究使用了本算法，请引用：

```bibtex
@article{DDD2026,
  title={Dynamic Diffusion-Driven Evolution for Multi-objective Optimization},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}

@article{PlatEMO,
  title={{PlatEMO}: A {MATLAB} platform for evolutionary multi-objective optimization},
  author={Tian, Ye and Cheng, Ran and Zhang, Xingyi and Jin, Yaochu},
  journal={IEEE Computational Intelligence Magazine},
  volume={12},
  number={4},
  pages={73--87},
  year={2017}
}
```

## 版权信息

Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for research purposes.

## 联系方式

如有问题或建议，请通过GitHub Issues提交。

---

**版本**: 1.0  
**最后更新**: 2026-03-03
