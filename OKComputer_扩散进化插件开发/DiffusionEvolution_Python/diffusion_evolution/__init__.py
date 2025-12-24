"""
DiffusionEvolution: 扩散模型与进化算法混合的多目标优化算法

这是一个将扩散模型与传统进化算法相结合的多目标优化插件，
专为PlatEMO平台设计，支持通过MATLAB-Python接口调用。

主要功能:
    - 扩散模型作为智能变异算子
    - 混合进化策略（扩散生成 + 传统遗传操作）
    - 支持多目标和多约束优化
    - 条件生成（基于适应度或排序信息）
    
作者: AI Assistant
日期: 2025
版本: 1.0.0
"""

from .core import DiffusionEvolution
from .diffusion import DiffusionModel, NoiseScheduler
from .selection import EnvironmentalSelection, TournamentSelection
from .operators import OperatorGA, SBX, PolynomialMutation
from .utils import normalize, denormalize, calculate_diversity

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "developer@example.com"

__all__ = [
    "DiffusionEvolution",
    "DiffusionModel", 
    "NoiseScheduler",
    "EnvironmentalSelection",
    "TournamentSelection",
    "OperatorGA",
    "normalize",
    "denormalize",
    "calculate_diversity"
]