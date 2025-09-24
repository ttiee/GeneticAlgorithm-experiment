"""
遗传算法实验二主程序
求解作业车间调度问题：3个工件在3台机器上的调度优化

题干：
3个工件(j1, j2, j3)需要在3台机器(m1, m2, m3)上进行加工
每个工件分别有3个工序，各工序加工所需的机器及其加工时间如下:
| 工件 | 工序1时间 | 工序2时间 | 工序3时间 | 工序1机器 | 工序2机器 | 工序3机器 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| j1   | 3         | 9         | 2         | m1        | m2        | m3        |
| j2   | 1         | 5         | 7         | m1        | m3        | m2        |
| j3   | 3         | 2         | 3         | m2        | m1        | m3        |
重要约束:
每个工件的工序必须按照预定的顺序执行
每台机器同时只能加工一个工序
目标：最小化最大完工时间(makespan)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import copy
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Operation:
    """工序类"""
    job_id: int
    operation_id: int
    machine_id: int
    processing_time: int
    start_time: int = 0
    end_time: int = 0


@dataclass
class Job:
    """工件类"""
    job_id: int
    operations: List[Operation]
    completion_time: int = 0


@dataclass
class Machine:
    """机器类"""
    machine_id: int
    operations: List[Operation]
    completion_time: int = 0


class JobShopScheduler:
    """作业车间调度器"""
    
    def __init__(self):
        """初始化调度器"""
        self.jobs = []
        self.machines = []
        self.operations = []
        self.makespan = 0
        
    def create_problem_instance(self):
        """创建实验2的问题实例"""
        # 清空现有数据
        self.jobs = []
        self.machines = []
        self.operations = []
        
        # 创建3台机器
        for i in range(3):
            self.machines.append(Machine(machine_id=i+1, operations=[]))
        
        # 创建3个工件及其工序
        # 工件1: 工序1(m1,3) -> 工序2(m2,9) -> 工序3(m3,2)
        job1_ops = [
            Operation(1, 1, 1, 3),  # 工件1，工序1，机器1，时间3
            Operation(1, 2, 2, 9),  # 工件1，工序2，机器2，时间9
            Operation(1, 3, 3, 2)   # 工件1，工序3，机器3，时间2
        ]
        self.jobs.append(Job(1, job1_ops))
        
        # 工件2: 工序1(m1,1) -> 工序2(m3,5) -> 工序3(m2,7)
        job2_ops = [
            Operation(2, 1, 1, 1),  # 工件2，工序1，机器1，时间1
            Operation(2, 2, 3, 5),  # 工件2，工序2，机器3，时间5
            Operation(2, 3, 2, 7)   # 工件2，工序3，机器2，时间7
        ]
        self.jobs.append(Job(2, job2_ops))
        
        # 工件3: 工序1(m2,3) -> 工序2(m1,2) -> 工序3(m3,3)
        job3_ops = [
            Operation(3, 1, 2, 3),  # 工件3，工序1，机器2，时间3
            Operation(3, 2, 1, 2),  # 工件3，工序2，机器1，时间2
            Operation(3, 3, 3, 3)   # 工件3，工序3，机器3，时间3
        ]
        self.jobs.append(Job(3, job3_ops))
        
        # 收集所有工序
        for job in self.jobs:
            self.operations.extend(job.operations)
    
    def validate_schedule_constraints(self, schedule: List[int]) -> bool:
        """
        验证调度方案是否满足工序顺序约束
        
        Args:
            schedule: 调度序列
            
        Returns:
            是否满足约束
        """
        # 记录每个工件的工序执行顺序
        job_operation_order = {1: [], 2: [], 3: []}
        
        for operation_id in schedule:
            operation = self.operations[operation_id]
            job_id = operation.job_id
            operation_seq = operation.operation_id
            job_operation_order[job_id].append(operation_seq)
        
        # 检查每个工件的工序是否按顺序执行
        for job_id in [1, 2, 3]:
            operations = job_operation_order[job_id]
            if operations != sorted(operations):
                return False
        
        return True
    
    def calculate_makespan(self, schedule: List[int]) -> int:
        """
        计算给定调度方案的最大完工时间
        
        Args:
            schedule: 调度序列，包含所有工序的排列
            
        Returns:
            最大完工时间（makespan）
        """
        # 首先验证工序顺序约束
        if not self.validate_schedule_constraints(schedule):
            # 如果违反约束，返回一个很大的惩罚值
            return 999999
        
        # 重置所有时间和状态
        for job in self.jobs:
            job.completion_time = 0
            for op in job.operations:
                op.start_time = 0
                op.end_time = 0
        
        for machine in self.machines:
            machine.completion_time = 0
            machine.operations = []
        
        # 按照调度序列执行工序
        current_time = 0
        
        for operation_id in schedule:
            operation = self.operations[operation_id]
            job = self.jobs[operation.job_id - 1]
            machine = self.machines[operation.machine_id - 1]
            
            # 计算工序开始时间
            # 必须等待前一道工序完成
            prev_operation_end = 0
            if operation.operation_id > 1:
                prev_operation = job.operations[operation.operation_id - 2]
                prev_operation_end = prev_operation.end_time
            
            # 必须等待机器空闲
            machine_available_time = machine.completion_time
            
            # 工序开始时间取两者最大值
            operation.start_time = max(prev_operation_end, machine_available_time)
            operation.end_time = operation.start_time + operation.processing_time
            
            # 更新机器状态
            machine.completion_time = operation.end_time
            machine.operations.append(operation)
            
            # 更新工件完成时间
            job.completion_time = max(job.completion_time, operation.end_time)
            
            current_time = max(current_time, operation.end_time)
        
        self.makespan = current_time
        return current_time
    
    def get_schedule_info(self, schedule: List[int]) -> Dict:
        """
        获取调度方案的详细信息
        
        Args:
            schedule: 调度序列
            
        Returns:
            包含详细信息的字典
        """
        makespan = self.calculate_makespan(schedule)
        
        # 收集每个工件的完成时间
        job_completion_times = {}
        for job in self.jobs:
            job_completion_times[job.job_id] = job.completion_time
        
        # 收集每台机器的完成时间
        machine_completion_times = {}
        for machine in self.machines:
            machine_completion_times[machine.machine_id] = machine.completion_time
        
        return {
            'makespan': makespan,
            'job_completion_times': job_completion_times,
            'machine_completion_times': machine_completion_times,
            'schedule': schedule
        }
    
    def print_schedule(self, schedule: List[int]):
        """打印调度方案"""
        info = self.get_schedule_info(schedule)
        
        print(f"\n调度方案 (最大完工时间: {info['makespan']}):")
        print("=" * 60)
        
        # 按时间顺序显示工序
        operation_times = []
        for i, op_id in enumerate(schedule):
            operation = self.operations[op_id]
            operation_times.append((
                operation.job_id, 
                operation.operation_id, 
                operation.machine_id,
                operation.start_time,
                operation.end_time,
                operation.processing_time
            ))
        
        # 按开始时间排序
        operation_times.sort(key=lambda x: x[3])
        
        print("时间轴调度:")
        for job_id, op_id, machine_id, start, end, duration in operation_times:
            print(f"时间 {start:2d}-{end:2d}: 工件{job_id} 工序{op_id} 在机器{machine_id} 上加工 (耗时{duration})")
        
        print(f"\n各工件完成时间:")
        for job_id, completion_time in info['job_completion_times'].items():
            print(f"  工件{job_id}: {completion_time}")
        
        print(f"\n各机器完成时间:")
        for machine_id, completion_time in info['machine_completion_times'].items():
            print(f"  机器{machine_id}: {completion_time}")


class JobShopIndividual:
    """作业车间调度问题的个体类"""
    
    def __init__(self, chromosome: List[int] = None, num_operations: int = 9):
        """
        初始化个体
        
        Args:
            chromosome: 染色体（工序排列）
            num_operations: 工序总数
        """
        if chromosome is None:
            # 生成随机调度序列
            self.chromosome = list(range(num_operations))
            random.shuffle(self.chromosome)
        else:
            self.chromosome = chromosome.copy()
        
        self.fitness = 0.0
        self.makespan = 0
    
    def calculate_fitness(self, scheduler: JobShopScheduler) -> float:
        """计算适应度（makespan的倒数，因为我们要最小化makespan）"""
        self.makespan = scheduler.calculate_makespan(self.chromosome)
        # 使用倒数作为适应度，makespan越小适应度越高
        self.fitness = 1.0 / (self.makespan + 1)  # 加1避免除零
        return self.fitness
    
    def __str__(self):
        return f"JobShopIndividual(chromosome={self.chromosome}, makespan={self.makespan}, fitness={self.fitness:.4f})"


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


class JobShopVisualizer:
    """作业车间调度问题可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.fig_size = (15, 10)
        self.dpi = 100
    
    def plot_evolution_process(self, 
                             best_fitness_history: list,
                             average_fitness_history: list,
                             save_path: str = None):
        """
        绘制进化过程图
        
        Args:
            best_fitness_history: 最佳适应度历史
            average_fitness_history: 平均适应度历史
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 8), dpi=self.dpi)
        
        generations = list(range(1, len(best_fitness_history) + 1))
        
        plt.plot(generations, best_fitness_history, 'r-', linewidth=3, 
                label='最佳适应度', marker='o', markersize=6)
        plt.plot(generations, average_fitness_history, 'b-', linewidth=3, 
                label='平均适应度', marker='s', markersize=6)
        
        plt.fill_between(generations, average_fitness_history, best_fitness_history, 
                        alpha=0.2, color='green', label='适应度范围')
        
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('适应度值', fontsize=12)
        plt.title('作业车间调度遗传算法进化过程', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加最优解信息
        if best_fitness_history:
            final_best = max(best_fitness_history)
            plt.annotate(f'最优适应度: {final_best:.4f}', 
                        xy=(len(generations), final_best), 
                        xytext=(len(generations) * 0.7, final_best * 1.1),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"进化过程图已保存到: {save_path}")
        
        plt.show()
    
    def plot_detailed_analysis(self, 
                             convergence_history: list,
                             save_path: str = None):
        """
        绘制详细分析图
        
        Args:
            convergence_history: 收敛历史详细信息
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size, dpi=self.dpi)
        fig.suptitle('作业车间调度遗传算法详细分析', fontsize=16, fontweight='bold')
        
        generations = [conv['generation'] for conv in convergence_history]
        best_fitnesses = [conv['best_fitness'] for conv in convergence_history]
        avg_fitnesses = [conv['average_fitness'] for conv in convergence_history]
        best_makespans = [conv['best_makespan'] for conv in convergence_history]
        
        # 1. 适应度对比
        ax1 = axes[0, 0]
        ax1.plot(generations, best_fitnesses, 'r-', linewidth=2, label='最佳适应度')
        ax1.plot(generations, avg_fitnesses, 'b-', linewidth=2, label='平均适应度')
        ax1.fill_between(generations, avg_fitnesses, best_fitnesses, alpha=0.3, color='green')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('适应度值')
        ax1.set_title('适应度对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 完工时间变化
        ax2 = axes[0, 1]
        ax2.plot(generations, best_makespans, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('完工时间')
        ax2.set_title('最优完工时间变化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 收敛性分析
        ax3 = axes[1, 0]
        if len(best_fitnesses) > 1:
            convergence_rate = np.diff(best_fitnesses)
            ax3.plot(generations[1:], convergence_rate, 'purple', linewidth=2, marker='^')
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('适应度变化')
            ax3.set_title('收敛性分析')
            ax3.grid(True, alpha=0.3)
        
        # 4. 统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 计算统计信息
        final_best = convergence_history[-1]['best_fitness']
        initial_best = convergence_history[0]['best_fitness']
        improvement = final_best - initial_best
        improvement_rate = (improvement / abs(initial_best)) * 100 if initial_best != 0 else 0
        
        final_makespan = convergence_history[-1]['best_makespan']
        initial_makespan = convergence_history[0]['best_makespan']
        makespan_improvement = initial_makespan - final_makespan
        
        stats_text = f"""
        算法统计信息:
        
        初始最佳适应度: {initial_best:.4f}
        最终最佳适应度: {final_best:.4f}
        适应度改进: {improvement:.4f}
        改进率: {improvement_rate:.2f}%
        
        初始完工时间: {initial_makespan}
        最终完工时间: {final_makespan}
        完工时间改进: {makespan_improvement}
        
        总迭代次数: {len(convergence_history)}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_gantt_chart(self, 
                        schedule: list,
                        scheduler: JobShopScheduler,
                        save_path: str = None):
        """
        绘制甘特图
        
        Args:
            schedule: 调度序列
            scheduler: 调度器
            save_path: 保存路径
        """
        # 计算调度信息
        info = scheduler.get_schedule_info(schedule)
        
        # 准备数据
        operations = []
        for i, op_id in enumerate(schedule):
            operation = scheduler.operations[op_id]
            operations.append({
                'job_id': operation.job_id,
                'operation_id': operation.operation_id,
                'machine_id': operation.machine_id,
                'start_time': operation.start_time,
                'end_time': operation.end_time,
                'processing_time': operation.processing_time
            })
        
        # 按开始时间排序
        operations.sort(key=lambda x: x['start_time'])
        
        # 创建甘特图
        fig, ax = plt.subplots(figsize=(15, 8), dpi=self.dpi)
        
        # 颜色映射
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # 为每台机器绘制工序
        machine_y_positions = {1: 0, 2: 1, 3: 2}
        machine_names = {1: '机器1', 2: '机器2', 3: '机器3'}
        
        for i, op in enumerate(operations):
            machine_id = op['machine_id']
            y_pos = machine_y_positions[machine_id]
            
            # 绘制工序条
            color = colors[op['job_id'] - 1]
            ax.barh(y_pos, op['processing_time'], 
                   left=op['start_time'], height=0.6, 
                   color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            # 添加标签
            ax.text(op['start_time'] + op['processing_time']/2, y_pos, 
                   f'J{op["job_id"]}O{op["operation_id"]}', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 设置坐标轴
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['机器1', '机器2', '机器3'])
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('机器', fontsize=12)
        ax.set_title(f'作业车间调度甘特图 (完工时间: {info["makespan"]})', 
                    fontsize=14, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        legend_elements = []
        for i in range(3):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                               facecolor=colors[i], 
                                               label=f'工件{i+1}'))
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 设置x轴范围
        ax.set_xlim(0, info['makespan'] + 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"甘特图已保存到: {save_path}")
        
        plt.show()


def run_experiment():
    """运行作业车间调度遗传算法实验"""
    print("=" * 60)
    print("遗传算法实验二: 作业车间调度问题")
    print("=" * 60)
    
    # 显示问题描述
    print("问题描述:")
    print("3个工件(j1, j2, j3)需要在3台机器(m1, m2, m3)上进行加工")
    print("每个工件分别有3个工序，各工序加工所需的机器及其加工时间如下:")
    print()
    print("| 工件 | 工序1时间 | 工序2时间 | 工序3时间 | 工序1机器 | 工序2机器 | 工序3机器 |")
    print("|------|-----------|-----------|-----------|-----------|-----------|-----------|")
    print("| j1   | 3         | 9         | 2         | m1        | m2        | m3        |")
    print("| j2   | 1         | 5         | 7         | m1        | m3        | m2        |")
    print("| j3   | 3         | 2         | 3         | m2        | m1        | m3        |")
    print()
    print("重要约束:")
    print("✓ 每个工件的工序必须按照预定的顺序执行")
    print("  - 工件1: 工序1 → 工序2 → 工序3")
    print("  - 工件2: 工序1 → 工序2 → 工序3") 
    print("  - 工件3: 工序1 → 工序2 → 工序3")
    print("✓ 每台机器同时只能加工一个工序")
    print("✓ 目标：最小化最大完工时间(makespan)")
    print()
    
    # 算法参数配置
    config = {
        'population_size': 5,      # 种群大小
        'crossover_rate': 0.8,      # 交叉概率
        'mutation_rate': 0.1,       # 变异概率
        'max_generations': 100      # 最大迭代次数
    }
    
    print("算法参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建遗传算法实例
    ga = JobShopGeneticAlgorithm(
        population_size=config['population_size'],
        crossover_rate=config['crossover_rate'],
        mutation_rate=config['mutation_rate'],
        max_generations=config['max_generations']
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行遗传算法
    best_solution, best_fitness_history, avg_fitness_history, convergence_history = ga.run()
    
    # 记录结束时间
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n算法执行时间: {execution_time:.2f} 秒")
    print(f"最优解: 完工时间 = {best_solution.makespan}")
    
    # 显示最优调度方案
    scheduler = ga.get_scheduler()
    scheduler.print_schedule(best_solution.chromosome)
    
    # 创建可视化器
    visualizer = JobShopVisualizer()
    
    # 创建输出目录
    output_dir = "results-2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 绘制进化过程图
    print("\n正在生成可视化图表...")
    
    # 1. 进化过程图
    visualizer.plot_evolution_process(
        best_fitness_history, 
        avg_fitness_history,
        save_path=os.path.join(output_dir, "evolution_process.png")
    )
    
    # 2. 详细分析图
    visualizer.plot_detailed_analysis(
        convergence_history,
        save_path=os.path.join(output_dir, "detailed_analysis.png")
    )
    
    # 3. 甘特图
    visualizer.plot_gantt_chart(
        best_solution.chromosome,
        scheduler,
        save_path=os.path.join(output_dir, "gantt_chart.png")
    )
    
    print(f"\n所有图表已保存到 '{output_dir}' 目录")
    
    return {
        'best_solution': best_solution,
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'convergence_history': convergence_history,
        'execution_time': execution_time,
        'scheduler': scheduler
    }


def run_multiple_experiments(num_runs=5):
    """
    运行多次实验以分析算法稳定性
    
    Args:
        num_runs: 实验次数
    """
    print(f"\n运行 {num_runs} 次实验以分析算法稳定性...")
    
    results = []
    config = {
        'population_size': 50,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'max_generations': 100
    }
    
    for run in range(num_runs):
        print(f"\n第 {run + 1} 次实验:")
        print("-" * 30)
        
        ga = JobShopGeneticAlgorithm(
            population_size=config['population_size'],
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate'],
            max_generations=config['max_generations']
        )
        
        best_solution, _, _, _ = ga.run()
        results.append({
            'run': run + 1,
            'best_makespan': best_solution.makespan,
            'best_fitness': best_solution.fitness
        })
    
    # 分析结果
    print("\n" + "=" * 50)
    print("多次实验结果:")
    print("=" * 50)
    
    best_makespans = [r['best_makespan'] for r in results]
    best_fitnesses = [r['best_fitness'] for r in results]
    
    print(f"最佳完工时间统计:")
    print(f"  平均值: {np.mean(best_makespans):.2f}")
    print(f"  标准差: {np.std(best_makespans):.2f}")
    print(f"  最小值: {np.min(best_makespans):.2f}")
    print(f"  最大值: {np.max(best_makespans):.2f}")
    
    print(f"\n最佳适应度统计:")
    print(f"  平均值: {np.mean(best_fitnesses):.4f}")
    print(f"  标准差: {np.std(best_fitnesses):.4f}")
    print(f"  最大值: {np.max(best_fitnesses):.4f}")
    print(f"  最小值: {np.min(best_fitnesses):.4f}")
    
    # 找到最佳解
    best_run = min(results, key=lambda x: x['best_makespan'])
    print(f"\n最佳解:")
    print(f"  第 {best_run['run']} 次实验")
    print(f"  完工时间: {best_run['best_makespan']}")
    print(f"  适应度: {best_run['best_fitness']:.4f}")


if __name__ == "__main__":
    # 运行单次实验
    results = run_experiment()
    
    # 询问是否运行多次实验
    print("\n是否运行多次实验以分析算法稳定性? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', '是']:
            run_multiple_experiments(5)
    except:
        print("跳过多次实验")
    
    print("\n实验完成!")
