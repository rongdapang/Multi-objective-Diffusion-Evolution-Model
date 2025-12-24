"""
扩散-进化模型 (Diffusion-Evolution Model)

一个结合了扩散模型和进化算法的深度学习框架，
用于生成高质量的数据样本并优化模型性能。
"""

__version__ = "1.0.0"
__author__ = "Diffusion-Evolution Team"
__email__ = "contact@diffusion-evolution.com"
__description__ = "A framework combining Diffusion Models and Evolutionary Algorithms"

# 导入主要组件
from .models.diffusion import DiffusionModel, UNet
from .models.evolution import (
    GeneticAlgorithm,
    TournamentSelection,
    RouletteWheelSelection,
    RankSelection,
    DiffusionEvolutionOptimizer
)
from .training.trainer import DiffusionEvolutionTrainer
from .configs.config import ConfigManager, ModelConfig, TrainingConfig, EvolutionConfig

# 定义公开的API
__all__ = [
    # 模型
    'DiffusionModel',
    'UNet',
    
    # 进化算法
    'GeneticAlgorithm',
    'TournamentSelection',
    'RouletteWheelSelection',
    'RankSelection',
    'DiffusionEvolutionOptimizer',
    
    # 训练器
    'DiffusionEvolutionTrainer',
    
    # 配置
    'ConfigManager',
    'ModelConfig',
    'TrainingConfig',
    'EvolutionConfig',
]

# 版本信息
def get_version():
    """获取版本信息"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__
    }

# 检查依赖
def check_dependencies():
    """检查必要的依赖是否安装"""
    missing_deps = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, will use CPU")
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    return True

# 初始化检查
if __name__ != "__main__":
    check_dependencies()