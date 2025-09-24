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
        self.individuals = self._create_constraint_satisfying_population(size, num_operations)
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.best_fitness_history = []
        self.average_fitness_history = []
    
    def _create_constraint_satisfying_population(self, size: int, num_operations: int) -> List[JobShopIndividual]:
        """
        创建满足工序顺序约束的初始种群
        
        Args:
            size: 种群大小
            num_operations: 工序总数
            
        Returns:
            满足约束的个体列表
        """
        individuals = []
        
        for _ in range(size):
            # 为每个工件创建工序序列
            job1_ops = [0, 1, 2]  # 工件1的工序：0,1,2
            job2_ops = [3, 4, 5]  # 工件2的工序：3,4,5  
            job3_ops = [6, 7, 8]  # 工件3的工序：6,7,8
            
            # 随机打乱每个工件的工序顺序，但保持工件内工序顺序
            random.shuffle(job1_ops)
            random.shuffle(job2_ops)
            random.shuffle(job3_ops)
            
            # 将所有工序合并成一个调度序列
            all_operations = job1_ops + job2_ops + job3_ops
            random.shuffle(all_operations)
            
            # 创建满足约束的调度序列
            schedule = self._create_valid_schedule(all_operations)
            
            individuals.append(JobShopIndividual(schedule))
        
        return individuals
    
    def _create_valid_schedule(self, operations: List[int]) -> List[int]:
        """
        创建满足工序顺序约束的调度序列
        
        Args:
            operations: 工序列表
            
        Returns:
            满足约束的调度序列
        """
        # 工序到工件的映射
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        # 按工件分组工序
        job_operations = {1: [], 2: [], 3: []}
        for op in operations:
            job_id = operation_to_job[op]
            job_operations[job_id].append(op)
        
        # 对每个工件的工序按顺序排序
        for job_id in [1, 2, 3]:
            job_operations[job_id].sort(key=lambda x: operation_to_seq[x])
        
        # 创建满足约束的调度序列
        valid_schedule = []
        
        # 随机选择下一个要调度的工序
        remaining_ops = operations.copy()
        
        while remaining_ops:
            # 找到可以调度的工序（其前序工序已完成）
            available_ops = []
            
            for op in remaining_ops:
                job_id = operation_to_job[op]
                seq = operation_to_seq[op]
                
                # 检查前序工序是否已完成
                can_schedule = True
                for prev_seq in range(1, seq):
                    prev_op = None
                    for check_op in operations:
                        if operation_to_job[check_op] == job_id and operation_to_seq[check_op] == prev_seq:
                            prev_op = check_op
                            break
                    
                    if prev_op is not None and prev_op not in [x for x in operations if x not in remaining_ops]:
                        can_schedule = False
                        break
                
                if can_schedule:
                    available_ops.append(op)
            
            # 从可用工序中随机选择一个
            if available_ops:
                selected_op = random.choice(available_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
            else:
                # 如果没有可用工序，随机选择一个（这种情况理论上不应该发生）
                selected_op = random.choice(remaining_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
        
        return valid_schedule
    
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
        """交叉操作 - 使用保持工序顺序约束的交叉"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # 创建满足约束的子代
        child1_chromosome = self._create_constraint_satisfying_offspring(parent1.chromosome, parent2.chromosome)
        child2_chromosome = self._create_constraint_satisfying_offspring(parent2.chromosome, parent1.chromosome)
        
        return JobShopIndividual(child1_chromosome), JobShopIndividual(child2_chromosome)
    
    def _create_constraint_satisfying_offspring(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        创建满足工序顺序约束的后代
        
        Args:
            parent1: 父代1的染色体
            parent2: 父代2的染色体
            
        Returns:
            满足约束的后代染色体
        """
        # 工序到工件的映射
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        # 随机选择从哪个父代继承每个工件的工序顺序
        job_inheritance = {}
        for job_id in [1, 2, 3]:
            job_inheritance[job_id] = random.choice([parent1, parent2])
        
        # 按工件分组工序
        job_operations = {1: [], 2: [], 3: []}
        for op in range(9):
            job_id = operation_to_job[op]
            job_operations[job_id].append(op)
        
        # 从选定的父代中获取每个工件的工序顺序
        offspring_operations = []
        
        for job_id in [1, 2, 3]:
            parent = job_inheritance[job_id]
            
            # 从父代中提取该工件的工序顺序
            job_ops_in_parent = []
            for op in parent:
                if operation_to_job[op] == job_id:
                    job_ops_in_parent.append(op)
            
            # 按工序顺序排序
            job_ops_in_parent.sort(key=lambda x: operation_to_seq[x])
            offspring_operations.extend(job_ops_in_parent)
        
        # 创建满足约束的调度序列
        return self._create_valid_schedule_from_operations(offspring_operations)
    
    def _create_valid_schedule_from_operations(self, operations: List[int]) -> List[int]:
        """
        从工序列表创建满足约束的调度序列
        
        Args:
            operations: 工序列表
            
        Returns:
            满足约束的调度序列
        """
        # 工序到工件的映射
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        # 创建满足约束的调度序列
        valid_schedule = []
        remaining_ops = operations.copy()
        
        while remaining_ops:
            # 找到可以调度的工序（其前序工序已完成）
            available_ops = []
            
            for op in remaining_ops:
                job_id = operation_to_job[op]
                seq = operation_to_seq[op]
                
                # 检查前序工序是否已完成
                can_schedule = True
                for prev_seq in range(1, seq):
                    prev_op = None
                    for check_op in operations:
                        if operation_to_job[check_op] == job_id and operation_to_seq[check_op] == prev_seq:
                            prev_op = check_op
                            break
                    
                    if prev_op is not None and prev_op not in [x for x in operations if x not in remaining_ops]:
                        can_schedule = False
                        break
                
                if can_schedule:
                    available_ops.append(op)
            
            # 从可用工序中随机选择一个
            if available_ops:
                selected_op = random.choice(available_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
            else:
                # 如果没有可用工序，随机选择一个
                selected_op = random.choice(remaining_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
        
        return valid_schedule
    
    def mutation(self, individual: JobShopIndividual) -> JobShopIndividual:
        """变异操作 - 保持工序顺序约束的变异"""
        if random.random() > self.mutation_rate:
            return copy.deepcopy(individual)
        
        # 创建满足约束的变异个体
        mutated_chromosome = self._create_constraint_satisfying_mutation(individual.chromosome)
        return JobShopIndividual(mutated_chromosome)
    
    def _create_constraint_satisfying_mutation(self, chromosome: List[int]) -> List[int]:
        """
        创建满足工序顺序约束的变异个体
        
        Args:
            chromosome: 原始染色体
            
        Returns:
            满足约束的变异染色体
        """
        # 工序到工件的映射
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        # 按工件分组工序
        job_operations = {1: [], 2: [], 3: []}
        for op in chromosome:
            job_id = operation_to_job[op]
            job_operations[job_id].append(op)
        
        # 对每个工件的工序按顺序排序
        for job_id in [1, 2, 3]:
            job_operations[job_id].sort(key=lambda x: operation_to_seq[x])
        
        # 随机选择变异类型
        mutation_type = random.choice(['swap_jobs', 'reorder_within_job', 'random_reschedule'])
        
        if mutation_type == 'swap_jobs':
            # 交换两个工件的调度顺序
            job1, job2 = random.sample([1, 2, 3], 2)
            job_operations[job1], job_operations[job2] = job_operations[job2], job_operations[job1]
        
        elif mutation_type == 'reorder_within_job':
            # 在工件内部重新排序（保持工序顺序）
            job_id = random.choice([1, 2, 3])
            # 工件内部工序必须按顺序执行，所以这种变异实际上不改变顺序
            pass
        
        elif mutation_type == 'random_reschedule':
            # 随机重新调度，但保持工序顺序约束
            all_operations = job_operations[1] + job_operations[2] + job_operations[3]
            return self._create_valid_schedule_from_operations(all_operations)
        
        # 重新组合所有工序
        all_operations = job_operations[1] + job_operations[2] + job_operations[3]
        
        # 创建满足约束的调度序列
        return self._create_valid_schedule_from_operations(all_operations)
    
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
