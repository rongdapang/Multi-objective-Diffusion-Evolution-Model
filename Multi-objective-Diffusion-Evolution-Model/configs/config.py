import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # UNet 配置
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    time_emb_dim: int = 512
    
    # 扩散模型配置
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # 设备配置
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    
    # 数据配置
    image_size: int = 32
    num_workers: int = 4
    
    # 训练策略
    save_every: int = 10
    generate_every: int = 50
    evolution_interval: int = 100
    evolution_generations: int = 20
    
    # 优化器配置
    gradient_clip_val: float = 1.0
    
    # 目录配置
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    sample_dir: str = "./samples"


@dataclass
class EvolutionConfig:
    """进化算法配置"""
    # 种群配置
    population_size: int = 30
    gene_length: int = 100  # 将由模型自动设置
    
    # 遗传操作参数
    mutation_rate: float = 0.1
    mutation_strength: float = 0.01
    crossover_rate: float = 0.7
    elitism: int = 2
    
    # 选择策略
    selection_strategy: str = "tournament"  # "tournament", "roulette", "rank"
    tournament_size: int = 3
    
    # 进化参数
    num_generations: int = 50
    target_fitness: Optional[float] = None
    maximize: bool = True


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验名称
    experiment_name: str = "diffusion_evolution_experiment"
    
    # 随机种子
    seed: int = 42
    
    # 日志配置
    log_level: str = "INFO"
    
    # 数据配置
    dataset_name: str = "cifar10"  # "cifar10", "custom", etc.
    data_path: str = "./data"
    
    # 评估配置
    num_samples_to_generate: int = 64
    fid_evaluation: bool = False
    inception_score: bool = False


class ConfigManager:
    """配置管理器"""
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        experiment_config: Optional[ExperimentConfig] = None
    ):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.evolution_config = evolution_config or EvolutionConfig()
        self.experiment_config = experiment_config or ExperimentConfig()
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self) -> None:
        """创建必要的目录"""
        import os
        
        directories = [
            self.training_config.save_dir,
            self.training_config.log_dir,
            self.training_config.sample_dir,
            self.experiment_config.data_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_config(self, save_path: str) -> None:
        """保存配置到文件"""
        import json
        
        config_dict = {
            'model': self._dataclass_to_dict(self.model_config),
            'training': self._dataclass_to_dict(self.training_config),
            'evolution': self._dataclass_to_dict(self.evolution_config),
            'experiment': self._dataclass_to_dict(self.experiment_config)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_config(self, config_path: str) -> None:
        """从文件加载配置"""
        import json
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 更新配置（这里简化处理，实际应该更健壮）
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(self.training_config, key):
                    setattr(self.training_config, key, value)
        
        if 'evolution' in config_dict:
            for key, value in config_dict['evolution'].items():
                if hasattr(self.evolution_config, key):
                    setattr(self.evolution_config, key, value)
        
        if 'experiment' in config_dict:
            for key, value in config_dict['experiment'].items():
                if hasattr(self.experiment_config, key):
                    setattr(self.experiment_config, key, value)
    
    def _dataclass_to_dict(self, obj) -> dict:
        """将dataclass对象转换为字典"""
        result = {}
        for key, value in obj.__dict__.items():
            if hasattr(value, '__dict__'):  # 也是dataclass
                result[key] = self._dataclass_to_dict(value)
            else:
                # 处理torch.device类型
                if isinstance(value, torch.device):
                    result[key] = str(value)
                else:
                    result[key] = value
        return result
    
    def print_config(self) -> None:
        """打印配置信息"""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        
        print("\nModel Configuration:")
        for key, value in self.model_config.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nTraining Configuration:")
        for key, value in self.training_config.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nEvolution Configuration:")
        for key, value in self.evolution_config.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nExperiment Configuration:")
        for key, value in self.experiment_config.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)


def get_default_config() -> ConfigManager:
    """获取默认配置"""
    return ConfigManager()


def create_config_for_cifar10() -> ConfigManager:
    """为CIFAR-10数据集创建配置"""
    model_config = ModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        timesteps=1000
    )
    
    training_config = TrainingConfig(
        num_epochs=500,
        batch_size=64,
        image_size=32,
        save_every=20,
        generate_every=100
    )
    
    evolution_config = EvolutionConfig(
        population_size=20,
        num_generations=30,
        mutation_rate=0.05,
        mutation_strength=0.005
    )
    
    experiment_config = ExperimentConfig(
        experiment_name="diffusion_evolution_cifar10",
        dataset_name="cifar10"
    )
    
    return ConfigManager(model_config, training_config, evolution_config, experiment_config)


def create_config_for_custom_dataset(image_size: int = 64, channels: int = 3) -> ConfigManager:
    """为自定义数据集创建配置"""
    model_config = ModelConfig(
        in_channels=channels,
        out_channels=channels,
        base_channels=128,
        timesteps=1000
    )
    
    training_config = TrainingConfig(
        num_epochs=300,
        batch_size=32,
        image_size=image_size,
        save_every=10,
        generate_every=50
    )
    
    evolution_config = EvolutionConfig(
        population_size=15,
        num_generations=25
    )
    
    experiment_config = ExperimentConfig(
        experiment_name="diffusion_evolution_custom",
        dataset_name="custom"
    )
    
    return ConfigManager(model_config, training_config, evolution_config, experiment_config)