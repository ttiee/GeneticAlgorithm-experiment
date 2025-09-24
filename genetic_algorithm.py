"""
遗传算法核心模块
用于求解函数 y = x^5 - 1000 的最大值，其中 x 为 [0, 31] 间的整数
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import copy


class Individual:
    """个体类 - 表示遗传算法中的一个解"""
    
    def __init__(self, chromosome: List[int] = None, chromosome_length: int = 5):
        """
        初始化个体
        
        Args:
            chromosome: 染色体（二进制编码）
            chromosome_length: 染色体长度（5位二进制可表示0-31）
        """
        if chromosome is None:
            self.chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        else:
            self.chromosome = chromosome.copy()
        self.fitness = 0.0
        self.value = 0  # 解码后的x值
    
    def decode(self) -> int:
        """将二进制染色体解码为十进制值"""
        decimal = 0
        for i, bit in enumerate(self.chromosome):
            decimal += bit * (2 ** (len(self.chromosome) - 1 - i))
        self.value = decimal
        return decimal
    
    def calculate_fitness(self, objective_function) -> float:
        """计算适应度"""
        x = self.decode()
        self.fitness = objective_function(x)
        return self.fitness
    
    def __str__(self):
        return f"Individual(chromosome={self.chromosome}, value={self.value}, fitness={self.fitness:.2f})"


class Population:
    """种群类 - 管理个体集合"""
    
    def __init__(self, size: int, chromosome_length: int = 5):
        """
        初始化种群
        
        Args:
            size: 种群大小
            chromosome_length: 染色体长度
        """
        self.size = size
        self.chromosome_length = chromosome_length
        self.individuals = [Individual(chromosome_length=chromosome_length) for _ in range(size)]
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.best_fitness_history = []
        self.average_fitness_history = []
    
    def evaluate_population(self, objective_function):
        """评估整个种群的适应度"""
        for individual in self.individuals:
            individual.calculate_fitness(objective_function)
        
        # 更新最佳个体
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or self.individuals[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.individuals[0])
        
        # 记录统计信息
        fitnesses = [ind.fitness for ind in self.individuals]
        self.fitness_history.append(fitnesses)
        self.best_fitness_history.append(self.best_individual.fitness)
        self.average_fitness_history.append(np.mean(fitnesses))
    
    def get_best_individual(self) -> Individual:
        """获取当前最佳个体"""
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_average_fitness(self) -> float:
        """获取平均适应度"""
        return np.mean([ind.fitness for ind in self.individuals])


class GeneticAlgorithm:
    """遗传算法主类"""
    
    def __init__(self, 
                 population_size: int = 20,
                 chromosome_length: int = 5,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 max_generations: int = 50,
                 objective_function=None):
        """
        初始化遗传算法
        
        Args:
            population_size: 种群大小
            chromosome_length: 染色体长度
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            max_generations: 最大迭代次数
            objective_function: 目标函数
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.objective_function = objective_function or self.default_objective_function
        
        self.population = Population(population_size, chromosome_length)
        self.generation = 0
        self.convergence_history = []
    
    def default_objective_function(self, x: int) -> float:
        """默认目标函数: y = x^5 - 1000"""
        return x ** 5 - 1000
    
    def selection(self, population: Population) -> List[Individual]:
        """选择操作 - 使用轮盘赌选择"""
        total_fitness = sum(ind.fitness for ind in population.individuals)
        if total_fitness == 0:
            return random.choices(population.individuals, k=population.size)
        
        # 处理负适应度值
        min_fitness = min(ind.fitness for ind in population.individuals)
        if min_fitness < 0:
            adjusted_fitness = [ind.fitness - min_fitness + 1 for ind in population.individuals]
        else:
            adjusted_fitness = [ind.fitness for ind in population.individuals]
        
        total_adjusted = sum(adjusted_fitness)
        if total_adjusted == 0:
            return random.choices(population.individuals, k=population.size)
        
        probabilities = [f / total_adjusted for f in adjusted_fitness]
        return random.choices(population.individuals, weights=probabilities, k=population.size)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作 - 单点交叉"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        crossover_point = random.randint(1, self.chromosome_length - 1)
        
        child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        
        return Individual(child1_chromosome), Individual(child2_chromosome)
    
    def mutation(self, individual: Individual) -> Individual:
        """变异操作 - 位翻转变异"""
        mutated_chromosome = individual.chromosome.copy()
        
        for i in range(len(mutated_chromosome)):
            if random.random() < self.mutation_rate:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        
        return Individual(mutated_chromosome)
    
    def evolve_generation(self):
        """进化一代"""
        # 选择
        selected_individuals = self.selection(self.population)
        
        # 生成新种群
        new_individuals = []
        
        # 保留最佳个体（精英策略）
        if self.population.best_individual:
            new_individuals.append(copy.deepcopy(self.population.best_individual))
        
        # 交叉和变异
        for i in range(0, len(selected_individuals) - 1, 2):
            parent1, parent2 = selected_individuals[i], selected_individuals[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            new_individuals.extend([child1, child2])
        
        # 如果种群大小为奇数，添加一个随机个体
        while len(new_individuals) < self.population_size:
            new_individuals.append(Individual(chromosome_length=self.chromosome_length))
        
        # 更新种群
        self.population.individuals = new_individuals[:self.population_size]
        self.population.generation += 1
        self.generation += 1
    
    def run(self) -> Tuple[Individual, List[float], List[float], List[float]]:
        """运行遗传算法"""
        print("开始遗传算法优化...")
        print(f"种群大小: {self.population_size}")
        print(f"最大迭代次数: {self.max_generations}")
        print(f"交叉概率: {self.crossover_rate}")
        print(f"变异概率: {self.mutation_rate}")
        print("-" * 50)
        
        # 初始评估
        self.population.evaluate_population(self.objective_function)
        
        for generation in range(self.max_generations):
            # 进化一代
            self.evolve_generation()
            
            # 评估新种群
            self.population.evaluate_population(self.objective_function)
            
            # 记录收敛信息
            best_fitness = self.population.get_best_individual().fitness
            avg_fitness = self.population.get_average_fitness()
            self.convergence_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'average_fitness': avg_fitness,
                'best_individual': self.population.get_best_individual()
            })
            
            # 打印进度
            if (generation + 1) % 10 == 0 or generation == 0:
                best_ind = self.population.get_best_individual()
                print(f"第 {generation + 1:2d} 代: 最佳适应度 = {best_fitness:8.2f}, "
                      f"平均适应度 = {avg_fitness:8.2f}, 最佳个体 x = {best_ind.value}")
        
        print("-" * 50)
        print("优化完成!")
        best_solution = self.population.get_best_individual()
        print(f"最优解: x = {best_solution.value}, y = {best_solution.fitness:.2f}")
        
        return (best_solution, 
                self.population.best_fitness_history,
                self.population.average_fitness_history,
                self.convergence_history)
