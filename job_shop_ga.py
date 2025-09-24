"""
作业车间调度问题的遗传算法模块
专门用于求解作业车间调度问题
"""

import random
import numpy as np
from typing import List, Tuple, Optional
import copy
from job_shop_scheduling import JobShopScheduler, JobShopIndividual


class JobShopPopulation:
    """作业车间调度问题的种群类"""
    
    def __init__(self, size: int, num_operations: int = 9):
        """
        初始化种群
        
        Args:
            size: 种群大小
            num_operations: 工序总数
        """
        self.size = size
        self.num_operations = num_operations
        self.individuals = [JobShopIndividual(num_operations=num_operations) for _ in range(size)]
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.best_fitness_history = []
        self.average_fitness_history = []
    
    def evaluate_population(self, scheduler: JobShopScheduler):
        """评估整个种群的适应度"""
        for individual in self.individuals:
            individual.calculate_fitness(scheduler)
        
        # 更新最佳个体
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or self.individuals[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.individuals[0])
        
        # 记录统计信息
        fitnesses = [ind.fitness for ind in self.individuals]
        self.fitness_history.append(fitnesses)
        self.best_fitness_history.append(self.best_individual.fitness)
        self.average_fitness_history.append(np.mean(fitnesses))
    
    def get_best_individual(self) -> JobShopIndividual:
        """获取当前最佳个体"""
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_average_fitness(self) -> float:
        """获取平均适应度"""
        return np.mean([ind.fitness for ind in self.individuals])


class JobShopGeneticAlgorithm:
    """作业车间调度问题的遗传算法主类"""
    
    def __init__(self, 
                 population_size: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 max_generations: int = 100,
                 num_operations: int = 9):
        """
        初始化遗传算法
        
        Args:
            population_size: 种群大小
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            max_generations: 最大迭代次数
            num_operations: 工序总数
        """
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.num_operations = num_operations
        
        self.population = JobShopPopulation(population_size, num_operations)
        self.generation = 0
        self.convergence_history = []
        self.scheduler = JobShopScheduler()
        self.scheduler.create_problem_instance()
    
    def selection(self, population: JobShopPopulation) -> List[JobShopIndividual]:
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
    
    def crossover(self, parent1: JobShopIndividual, parent2: JobShopIndividual) -> Tuple[JobShopIndividual, JobShopIndividual]:
        """交叉操作 - 使用顺序交叉(OX)"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # 顺序交叉(Order Crossover)
        size = len(parent1.chromosome)
        start, end = sorted(random.sample(range(size), 2))
        
        # 创建子代1
        child1_chromosome = [-1] * size
        child1_chromosome[start:end] = parent1.chromosome[start:end]
        
        # 从parent2中填充剩余位置
        remaining = [x for x in parent2.chromosome if x not in child1_chromosome[start:end]]
        remaining_idx = 0
        
        for i in range(size):
            if child1_chromosome[i] == -1:
                child1_chromosome[i] = remaining[remaining_idx]
                remaining_idx += 1
        
        # 创建子代2
        child2_chromosome = [-1] * size
        child2_chromosome[start:end] = parent2.chromosome[start:end]
        
        # 从parent1中填充剩余位置
        remaining = [x for x in parent1.chromosome if x not in child2_chromosome[start:end]]
        remaining_idx = 0
        
        for i in range(size):
            if child2_chromosome[i] == -1:
                child2_chromosome[i] = remaining[remaining_idx]
                remaining_idx += 1
        
        return JobShopIndividual(child1_chromosome), JobShopIndividual(child2_chromosome)
    
    def mutation(self, individual: JobShopIndividual) -> JobShopIndividual:
        """变异操作 - 交换变异"""
        mutated_chromosome = individual.chromosome.copy()
        
        if random.random() < self.mutation_rate:
            # 随机选择两个位置进行交换
            pos1, pos2 = random.sample(range(len(mutated_chromosome)), 2)
            mutated_chromosome[pos1], mutated_chromosome[pos2] = mutated_chromosome[pos2], mutated_chromosome[pos1]
        
        return JobShopIndividual(mutated_chromosome)
    
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
            new_individuals.append(JobShopIndividual(num_operations=self.num_operations))
        
        # 更新种群
        self.population.individuals = new_individuals[:self.population_size]
        self.population.generation += 1
        self.generation += 1
    
    def run(self) -> Tuple[JobShopIndividual, List[float], List[float], List[dict]]:
        """运行遗传算法"""
        print("开始作业车间调度遗传算法优化...")
        print(f"种群大小: {self.population_size}")
        print(f"最大迭代次数: {self.max_generations}")
        print(f"交叉概率: {self.crossover_rate}")
        print(f"变异概率: {self.mutation_rate}")
        print("-" * 50)
        
        # 初始评估
        self.population.evaluate_population(self.scheduler)
        
        for generation in range(self.max_generations):
            # 进化一代
            self.evolve_generation()
            
            # 评估新种群
            self.population.evaluate_population(self.scheduler)
            
            # 记录收敛信息
            best_fitness = self.population.get_best_individual().fitness
            avg_fitness = self.population.get_average_fitness()
            best_makespan = self.population.get_best_individual().makespan
            
            self.convergence_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'average_fitness': avg_fitness,
                'best_makespan': best_makespan,
                'best_individual': self.population.get_best_individual()
            })
            
            # 打印进度
            if (generation + 1) % 10 == 0 or generation == 0:
                best_ind = self.population.get_best_individual()
                print(f"第 {generation + 1:3d} 代: 最佳适应度 = {best_fitness:8.4f}, "
                      f"平均适应度 = {avg_fitness:8.4f}, 最小完工时间 = {best_makespan:2d}")
        
        print("-" * 50)
        print("优化完成!")
        best_solution = self.population.get_best_individual()
        print(f"最优解: 完工时间 = {best_solution.makespan}, 适应度 = {best_solution.fitness:.4f}")
        
        return (best_solution, 
                self.population.best_fitness_history,
                self.population.average_fitness_history,
                self.convergence_history)
    
    def get_scheduler(self) -> JobShopScheduler:
        """获取调度器实例"""
        return self.scheduler


if __name__ == "__main__":
    # 测试遗传算法
    ga = JobShopGeneticAlgorithm(
        population_size=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=50
    )
    
    best_solution, best_fitness_history, avg_fitness_history, convergence_history = ga.run()
    
    print(f"\n最优调度方案:")
    print(f"调度序列: {best_solution.chromosome}")
    print(f"完工时间: {best_solution.makespan}")
    
    # 显示详细调度信息
    scheduler = ga.get_scheduler()
    scheduler.print_schedule(best_solution.chromosome)
