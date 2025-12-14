import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
import copy


class Individual:
    """个体类，用于进化算法"""
    
    def __init__(self, genes: torch.Tensor, fitness: Optional[float] = None):
        self.genes = genes.clone() if isinstance(genes, torch.Tensor) else torch.tensor(genes, dtype=torch.float32)
        self.fitness = fitness
        self.age = 0
    
    def clone(self) -> 'Individual':
        """克隆个体"""
        new_individual = Individual(self.genes, self.fitness)
        new_individual.age = self.age
        return new_individual
    
    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        """变异操作"""
        mask = torch.rand_like(self.genes) < mutation_rate
        noise = torch.randn_like(self.genes) * mutation_strength
        self.genes += mask * noise
    
    def crossover(self, other: 'Individual', crossover_rate: float = 0.5) -> 'Individual':
        """交叉操作"""
        if torch.rand(1).item() < crossover_rate:
            mask = torch.rand_like(self.genes) < 0.5
            child_genes = torch.where(mask, self.genes, other.genes)
            return Individual(child_genes)
        else:
            return self.clone()
    
    def __len__(self) -> int:
        return len(self.genes)


class Population:
    """种群类"""
    
    def __init__(self, individuals: Optional[List[Individual]] = None):
        self.individuals = individuals or []
        self.history = {
            'best_fitness': [],
            'average_fitness': [],
            'diversity': []
        }
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __getitem__(self, index) -> Individual:
        return self.individuals[index]
    
    def append(self, individual: Individual) -> None:
        self.individuals.append(individual)
    
    def extend(self, individuals: List[Individual]) -> None:
        self.individuals.extend(individuals)
    
    def remove(self, individual: Individual) -> None:
        self.individuals.remove(individual)
    
    def evaluate_fitness(self, fitness_func: Callable[[torch.Tensor], float]) -> None:
        """评估种群中所有个体的适应度"""
        for individual in self.individuals:
            if individual.fitness is None:
                try:
                    individual.fitness = fitness_func(individual.genes)
                except Exception as e:
                    print(f"Warning: Fitness evaluation failed for individual: {e}")
                    individual.fitness = float('-inf')
    
    def get_best_individual(self) -> Optional[Individual]:
        """获取最优个体"""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
    
    def get_worst_individual(self) -> Optional[Individual]:
        """获取最差个体"""
        if not self.individuals:
            return None
        return min(self.individuals, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
    
    def calculate_diversity(self) -> float:
        """计算种群多样性（基因的标准差）"""
        if len(self.individuals) < 2:
            return 0.0
        
        genes = torch.stack([ind.genes for ind in self.individuals])
        return torch.std(genes).item()
    
    def update_statistics(self) -> None:
        """更新统计信息"""
        if not self.individuals:
            return
        
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if fitnesses:
            self.history['best_fitness'].append(max(fitnesses))
            self.history['average_fitness'].append(np.mean(fitnesses))
            self.history['diversity'].append(self.calculate_diversity())
    
    def sort_by_fitness(self, reverse: bool = True) -> None:
        """按适应度排序"""
        self.individuals.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=reverse)


class SelectionStrategy(ABC):
    """选择策略抽象基类"""
    
    @abstractmethod
    def select(self, population: Population, num_parents: int) -> List[Individual]:
        """选择父代个体"""
        pass


class TournamentSelection(SelectionStrategy):
    """锦标赛选择"""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: Population, num_parents: int) -> List[Individual]:
        """选择父代"""
        parents = []
        for _ in range(num_parents):
            tournament = np.random.choice(population.individuals, self.tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
            parents.append(winner.clone())
        return parents


class RouletteWheelSelection(SelectionStrategy):
    """轮盘赌选择"""
    
    def select(self, population: Population, num_parents: int) -> List[Individual]:
        """选择父代"""
        fitnesses = np.array([ind.fitness for ind in population.individuals if ind.fitness is not None])
        if np.sum(fitnesses) == 0:
            probabilities = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            probabilities = fitnesses / np.sum(fitnesses)
        
        selected_indices = np.random.choice(len(population.individuals), num_parents, p=probabilities)
        return [population.individuals[i].clone() for i in selected_indices]


class RankSelection(SelectionStrategy):
    """排序选择"""
    
    def select(self, population: Population, num_parents: int) -> List[Individual]:
        """选择父代"""
        sorted_pop = sorted(population.individuals, key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
        n = len(sorted_pop)
        ranks = np.arange(1, n + 1)
        probabilities = ranks / np.sum(ranks)
        
        selected_indices = np.random.choice(n, num_parents, p=probabilities)
        return [sorted_pop[i].clone() for i in selected_indices]


class GeneticAlgorithm:
    """遗传算法"""
    
    def __init__(
        self,
        population_size: int = 50,
        gene_length: int = 10,
        fitness_function: Optional[Callable[[torch.Tensor], float]] = None,
        selection_strategy: SelectionStrategy = None,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        maximize: bool = True
    ):
        self.population_size = population_size
        self.gene_length = gene_length
        self.fitness_function = fitness_function
        self.selection_strategy = selection_strategy or TournamentSelection()
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.maximize = maximize
        
        self.population = None
        self.generation = 0
    
    def initialize_population(self) -> None:
        """初始化种群"""
        individuals = []
        for _ in range(self.population_size):
            genes = torch.randn(self.gene_length)
            individual = Individual(genes)
            individuals.append(individual)
        
        self.population = Population(individuals)
    
    def evolve_generation(self) -> None:
        """进化一代"""
        if self.population is None:
            self.initialize_population()
        
        # 评估适应度
        if self.fitness_function:
            self.population.evaluate_fitness(self.fitness_function)
        
        # 更新统计信息
        self.population.update_statistics()
        
        # 创建新一代
        new_population = Population()
        
        # 精英保留
        if self.elitism > 0:
            self.population.sort_by_fitness(reverse=self.maximize)
            elites = self.population.individuals[:self.elitism]
            new_population.extend([elite.clone() for elite in elites])
        
        # 选择、交叉、变异
        num_offspring = self.population_size - self.elitism
        parents = self.selection_strategy.select(self.population, num_offspring)
        
        # 生成后代
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                # 交叉
                child1 = parents[i].crossover(parents[i + 1], self.crossover_rate)
                child2 = parents[i + 1].crossover(parents[i], self.crossover_rate)
                
                # 变异
                child1.mutate(self.mutation_rate, self.mutation_strength)
                child2.mutate(self.mutation_rate, self.mutation_strength)
                
                offspring.extend([child1, child2])
            else:
                # 单个个体直接变异
                child = parents[i].clone()
                child.mutate(self.mutation_rate, self.mutation_strength)
                offspring.append(child)
        
        new_population.extend(offspring)
        
        # 确保种群大小正确
        if len(new_population) > self.population_size:
            new_population.individuals = new_population.individuals[:self.population_size]
        
        # 更新种群
        self.population = new_population
        self.generation += 1
        
        # 增加年龄
        for individual in self.population.individuals:
            individual.age += 1
    
    def run(self, num_generations: int, target_fitness: Optional[float] = None) -> Individual:
        """运行遗传算法"""
        for gen in range(num_generations):
            self.evolve_generation()
            
            best_individual = self.population.get_best_individual()
            if best_individual and target_fitness is not None:
                if self.maximize and best_individual.fitness >= target_fitness:
                    break
                elif not self.maximize and best_individual.fitness <= target_fitness:
                    break
            
            if gen % 10 == 0:
                avg_fitness = np.mean([ind.fitness for ind in self.population.individuals if ind.fitness is not None])
                print(f"Generation {gen}: Best Fitness = {best_individual.fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")
        
        return self.population.get_best_individual()
    
    def get_statistics(self) -> dict:
        """获取算法统计信息"""
        if not self.population:
            return {}
        
        fitnesses = [ind.fitness for ind in self.population.individuals if ind.fitness is not None]
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(fitnesses) if fitnesses else None,
            'average_fitness': np.mean(fitnesses) if fitnesses else None,
            'diversity': self.population.calculate_diversity()
        }


class DiffusionEvolutionOptimizer:
    """扩散模型进化优化器"""
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        gene_length: int = 100,
        population_size: int = 30,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.01,
        crossover_rate: float = 0.7,
        device: torch.device = None
    ):
        self.diffusion_model = diffusion_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 获取模型参数作为基因
        self.original_params = self._extract_model_params()
        gene_length = len(self.original_params)
        
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            gene_length=gene_length,
            fitness_function=self._evaluate_fitness,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            crossover_rate=crossover_rate,
            selection_strategy=TournamentSelection(tournament_size=3)
        )
    
    def _extract_model_params(self) -> torch.Tensor:
        """提取模型参数"""
        params = []
        for param in self.diffusion_model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def _apply_model_params(self, genes: torch.Tensor) -> None:
        """应用基因到模型参数"""
        start = 0
        for param in self.diffusion_model.parameters():
            end = start + param.numel()
            param.data = genes[start:end].view(param.shape).to(param.device)
            start = end
    
    def _evaluate_fitness(self, genes: torch.Tensor) -> float:
        """评估适应度（这里可以根据具体任务定义）"""
        try:
            # 应用基因到模型
            self._apply_model_params(genes)
            
            # 这里应该根据具体任务定义适应度函数
            # 例如：生成样本的质量、损失函数值等
            # 暂时返回一个模拟的适应度值
            return torch.randn(1).item()  # 占位符
        except Exception as e:
            print(f"Fitness evaluation error: {e}")
            return float('-inf')
    
    def optimize(self, num_generations: int = 50) -> dict:
        """优化模型"""
        print("Starting diffusion model optimization with genetic algorithm...")
        
        best_individual = self.ga.run(num_generations)
        
        if best_individual:
            self._apply_model_params(best_individual.genes)
            print(f"Optimization completed. Best fitness: {best_individual.fitness}")
        
        return self.ga.get_statistics()