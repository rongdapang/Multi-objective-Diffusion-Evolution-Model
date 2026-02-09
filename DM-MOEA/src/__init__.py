"""
DM-MOEA: 基于DDIM扩散模型引导的多目标进化算法
"""

from .core.individual import Individual
from .core.problem import Problem, ZDT1, ZDT2, ZDT3, DTLZ1, DTLZ2
from .operators.selection import fast_non_dominated_sort, compute_crowding_distance, environmental_selection
from .operators.genetic_operators import GeneticOperators
from .models.diffusion_model import DiffusionModel, ConditionEncoder
from .metrics.diversity import DiversityMetrics, AdaptiveDiversityThreshold
from .algorithm.dm_moea import DM_MOEA

__version__ = '1.0.0'
__all__ = [
    'Individual',
    'Problem',
    'ZDT1',
    'ZDT2',
    'ZDT3',
    'DTLZ1',
    'DTLZ2',
    'fast_non_dominated_sort',
    'compute_crowding_distance',
    'environmental_selection',
    'GeneticOperators',
    'DiffusionModel',
    'ConditionEncoder',
    'DiversityMetrics',
    'AdaptiveDiversityThreshold',
    'DM_MOEA'
]
