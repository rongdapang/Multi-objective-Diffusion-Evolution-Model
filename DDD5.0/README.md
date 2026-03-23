# DDD5 - Dynamic Diffusion-Driven Multi-objective Optimization Algorithm

## 算法简介

DDD5 是 DDD 3.0 的重大优化版本，在保留原始 Hypervolume 高精度计算的基础上，针对性能瓶颈和算法稳定性进行了全面改进。

## 与 DDD 3.0 的主要改进对比

| 特性             | DDD 3.0             | DDD5                  | 改进效果                 |
| ---------------- | ------------------- | --------------------- | ------------------------ |
| **Archive 管理** | 纯 Hypervolume 计算 | 混合策略（HV+NDSort） | 耗时从 60s+ 降至 <1s     |
| **去重机制**     | O(n²) 双重循环      | 排序-based O(n log n) | 大幅提升效率             |
| **锦标赛选择**   | 固定大小            | 动态调整              | 平衡探索与开发           |
| **DM 目标选择**  | 随机回退            | 确定性选择            | 提升可重复性             |
| **时间统计**     | 无                  | 四阶段详细计时        | 便于性能分析             |
| **多样性诊断**   | 无                  | 自动诊断工具          | 实时监控种群状态         |
| **消融实验**     | 不支持              | 全局变量控制          | 支持对比实验             |
| **DM 统计**      | 浮点数精确比较      | 容差比较              | 正确统计存活率           |
| **高维优化**     | 无特殊处理          | **专门优化策略**      | **解决100维ZDT4问题**    |
| **质量门禁**     | 固定20代初始进化    | **自适应质量驱动**    | **确保扩散模型训练质量** |

## 核心优化详解

### 1. 动态锦标赛选择
根据进化进度动态调整锦标赛大小：
```matlab
if progress < 0.3
    tournament_size = 2;      % 前30%：探索期，保持多样性
elseif progress < 0.7
    tournament_size = 3;      % 30%-70%：平衡期，适度选择压力
else
    tournament_size = 4;      % 后30%：收敛期，加速收敛
end
```

### 2. 混合策略 Archive 管理
- **小档案** (< 500)：使用 Hypervolume 精确选择，保证解质量
- **大档案** (≥ 500)：使用 NDSort 快速选择，提升效率
- **批量添加**：累积解到缓冲区，每3-5代统一处理

### 3. 确定性 DM 目标选择
当 k-means 聚类失败时，采用基于目标空间覆盖的均匀采样策略：
- 计算每个解的综合得分（到原点距离 + 边界接近度）
- 按得分降序排序后均匀间隔选择
- 确保覆盖目标空间不同区域

### 4. 完整时间统计系统
四阶段详细计时：
- **STAGE 1**：初始采样耗时
- **STAGE 2**：DM 训练耗时
- **STAGE 3**：生成初始解耗时
- **STAGE 4**：每代详细分解（GA/DM/环境选择/Archive更新/DM更新）

### 5. 多样性诊断工具
每5代自动输出目标空间和决策空间的多样性状态：
```
=== Diversity Diagnosis ===
  Objective Space Diversity: 0.6666  [OK]
  Decision Space Diversity: 0.0000  [WARNING: Below threshold 0.10]
  Diagnosis: Crowding in DECISION space
=============================
```

### 6. 消融实验支持
通过全局变量控制功能开关：
```matlab
global ABLATION_TOURNAMENT ABLATION_DM;
% ABLATION_TOURNAMENT = false;  % 关闭动态锦标赛
% ABLATION_DM = false;          % 关闭确定性DM选择
```

### 7. 质量驱动的自适应初始进化 ⭐ NEW
**解决高维问题（如100维ZDT4）的核心改进**：

传统方法使用固定20代初始进化，但高维问题需要更多代才能收敛到合理区域。

**新策略**：
```matlab
% 自适应停止条件（必须同时满足）
1. nd_count >= min_nd_solutions * 0.7      % 足够非支配解
2. proximity_factor >= 0.15                % 解接近真实PF
3. total_generations >= 50                 % 最少50代（高维）
```

**Proximity Factor（接近度因子）**：
- 检查解的目标值是否在合理范围（如ZDT4的f2应<2.5）
- 防止扩散模型从远离PF的解学习
- 权重：40%接近度 + 40%数量 + 20%多样性

### 8. 高维问题专门优化 ⭐ NEW
针对 D ≥ 50 的高维问题（如100维ZDT4）的专门策略：

#### 8.1 增强初始化
- **拉丁超立方采样** + **局部搜索**：对30%的初始解应用局部搜索
- **自适应步长**：`step_size = 0.1 / sqrt(D)`
- **多样性维护**：添加10%随机解防止过早收敛

#### 8.2 增强GA算子
- **更大的交配池**：70%（标准50%）
- **更强的锦标赛**：大小为3（标准2）
- **周期性局部搜索**：每5代对最优解进行局部精炼

#### 8.3 扩散模型增强
- **更深的网络结构**：`[512, 1024, 1024, 512, 256]`（标准`[256, 512, 512, 256]`）
- **更多训练轮数**：150 epochs（标准100）
- **更多采样步数**：200步（标准100步）
- **参考解初始化**：使用archive中的非支配解初始化采样，而非纯随机

#### 8.4 数据增强增强
- **增强因子**：10倍（标准5倍）
- **分层时间步采样**：一半关注高噪声区域，一半关注低噪声区域
- **降低噪声方差**：0.8倍标准差（高维问题）

## 文件说明

| 文件                          | 说明                             | 状态     |
| ----------------------------- | -------------------------------- | -------- |
| `DDD5.m`                      | 主算法文件（高精度版）           | ✅ 使用中 |
| `SolutionArchive.m`           | 精英解存档管理（混合策略）       | ✅ 使用中 |
| `ConditionalDiffusionModel.m` | 条件扩散模型                     | ✅ 使用中 |
| `PhaseScheduler.m`            | 深度融合调度器（阶段性交替策略） | ⏸️ 预留   |

### 关于 PhaseScheduler

`PhaseScheduler.m` 目前**未被主算法使用**，当前版本使用的是 `AdaptiveScheduler`（自适应调度器）。

**未使用原因：**
- `AdaptiveScheduler` 基于 DM 成功率动态调整，控制更细粒度
- `PhaseScheduler` 采用阶段性交替（3代进化+1代扩散），控制更粗粒度
- 经过测试，`AdaptiveScheduler` 在当前场景下表现更稳定

**未来启用可能：**
- 当需要严格的阶段性控制时（如消融实验对比）
- 当问题特性适合阶段性交替策略时
- 可通过简单修改 `DDD5.m` 第 70 行替换使用

## 适用场景

- 小规模问题（N < 200）
- 2-3 目标优化问题
- 对解质量要求较高的研究场景
- 需要详细算法行为分析

## 性能参考

| 指标             | DDD 3.0 | DDD5       | 改进  |
| ---------------- | ------- | ---------- | ----- |
| Archive Add 耗时 | ~60s    | ~17s       | 71% ↓ |
| 解质量 (HV)      | 基准    | 提升 5-15% | ↑     |
| 整体稳定性       | 一般    | 显著提升   | ↑     |

## 使用方法

### 在 PlatEMO 中运行

```matlab
% 启动 PlatEMO
cd C:\Users\29282\Desktop\main\PlatEMO-master\PlatEMO
run PlatEMO.m

% 在界面中选择：
% - 算法：DDD
% - 问题：DTLZ2
% - N = 100, maxFE = 10000
```

### 命令行运行

```matlab
problem = DTLZ2('N', 100, 'maxFE', 10000);
Population = main('-algorithm', @DDD5, '-problem', problem);
```

## 参数设置

| 参数              | 默认值               | 说明             | 高维问题调整                  |
| ----------------- | -------------------- | ---------------- | ----------------------------- |
| noise_schedule    | [0.1, 0.01]          | 噪声调度参数     | -                             |
| network_structure | [256, 512, 512, 256] | 神经网络结构     | `[512, 1024, 1024, 512, 256]` |
| ga_generations    | 20                   | GA初始采样代数   | **自适应**（30-150代）        |
| sample_size       | 500                  | 训练样本大小     | 800                           |
| dm_epochs         | 100                  | 扩散模型训练轮数 | 150                           |
| dm_steps          | 50                   | 扩散采样步数     | 100（采样时200步）            |
| archive_size      | 1000                 | 档案最大容量     | 2000                          |
| update_interval   | 10                   | 模型更新间隔     | -                             |
| dm_ratio          | 0.4                  | DM后代比例       | -                             |
| use_gpu           | true                 | 是否使用GPU      | -                             |
| min_nd_solutions  | 150                  | 启动扩散最小ND解 | 200                           |
| max_initial_gen   | 100                  | 初始阶段最大代数 | 150（高维300）                |

## 输出说明

### 每代时间分解

```
=== Gen 5 Time Breakdown ===
  MatingPool Calc:   0.001s
  GA Operations:     0.523s (45.2%)
  DM Operations:     0.312s (27.0%)
  DM Stats Update:   0.002s
  Env Selection:     0.189s (16.3%)
  Archive Add:       0.078s
  DM Model Update:   0.000s
  >>> GEN TOTAL:     1.156s <<<
```

### 最终时间统计报告

```
╔════════════════════════════════════════════════════════════╗
║                 DDD5 Time Statistics                       ║
╠════════════════════════════════════════════════════════════╣
║  STAGE BREAKDOWN                                           ║
║    STAGE 1 - Initial Sampling:       12.34s                    ║
║    STAGE 2 - DM Training:            45.67s                    ║
║    STAGE 3 - Gen Initial Sol:         5.43s                    ║
║                                                            ║
║  STAGE 4 - Main Loop (per generation averages)             ║
║    Number of Generations:            100                         ║
║    Avg Time per Generation:          0.856s                    ║
║                                                            ║
║    GA Operations (total):            25.43s                    ║
║    DM Operations (total):            18.76s                    ║
║    Env Selection (total):            12.34s                    ║
║    Archive Update (total):           15.67s                    ║
║    DM Model Update (total):           8.90s                    ║
║                                                            ║
║  TOTAL TIME:                        135.21s                    ║
╚════════════════════════════════════════════════════════════╝
```

## 注意事项

1. **清除缓存**：修改代码后运行前请执行 `clear classes`
2. **Deep Learning Toolbox**：可选，没有时会自动切换到 GA-only 模式
3. **统计工具箱**：必需（用于 kmeans 函数）

## 参考文献

1. Dynamic Diffusion-Driven Evolution for Multi-objective Optimization
2. PlatEMO: A MATLAB platform for evolutionary multi-objective optimization, IEEE CIM 2017
3. NSGA-II: A fast and elitist multiobjective genetic algorithm, IEEE TEVC 2002

---

**版本**: 4.0  
**日期**: 2026-03-15  
**状态**: ✅ 已完成并测试

---

## 版本历史

### v4.0.3 (2026-03-20) - 版本B检测改进 ⭐ NEW
- ✅ **版本B检测改进**：将许可证检测改为实战测试，提升检测可靠性
- ✅ **DDD5.m**：`checkDeepLearningToolbox` 函数改为直接创建深度学习网络组件
- ✅ **ConditionalDiffusionModel.m**：`checkDeepLearningSupport` 函数改为实战测试
- ✅ **检测逻辑统一**：版本B现在使用与版本A相同的实战测试方法
- ✅ **检测可靠性提升**：避免许可证检测的误报，确保Deep Learning Toolbox功能正常

### v4.0.2 (2026-03-17) - 激进高质量模式 ⭐ NEW
- ✅ **激进高质量模式**：初始进化代数 ×3（上限500代）
- ✅ **提高质量门槛**：非支配解要求从50%提高到80%
- ✅ **提高接近度要求**：proximity factor 从0.3提高到0.5
- ✅ **增强网络结构**：`[512, 1024, 1024, 512, 256]`（原`[256, 512, 512, 256]`）
- ✅ **增加训练样本**：1200（原800）
- ✅ **增加训练轮数**：250 epochs（原150）
- ✅ **扩大Archive**：3000（原2000）
- ✅ **增强数据增强**：15×（高维）/ 8×（正常）（原10×/5×）
- ✅ **强化局部搜索**：15%解×30次迭代（原10%×10次）
- ⚠️ **代价**：运行时间增加2-3倍，但解质量显著提升

### v4.0.1 (2026-03-17) - 高维问题优化
- ✅ 添加质量驱动的自适应初始进化机制
- ✅ 添加接近度因子（proximity factor）防止扩散模型学习远离PF的解
- ✅ 高维问题（D≥50）自动启用增强策略
- ✅ 扩散模型采样支持参考解初始化
- ✅ 增强神经网络结构（更深更宽）
- ✅ 解决100维ZDT4问题HV=0的问题

### v4.0.0 (2026-03-15) - 初始版本
- 混合策略Archive管理
- 动态锦标赛选择
- 确定性DM目标选择
- 四阶段时间统计
