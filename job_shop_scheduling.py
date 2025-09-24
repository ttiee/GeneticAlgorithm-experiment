"""
作业车间调度问题模块
用于求解3个工件在3台机器上的调度问题
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy


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


def create_random_schedule(num_operations: int = 9) -> List[int]:
    """创建随机调度序列"""
    schedule = list(range(num_operations))
    random.shuffle(schedule)
    return schedule


def validate_schedule(schedule: List[int], num_operations: int = 9) -> bool:
    """验证调度序列的有效性"""
    if len(schedule) != num_operations:
        return False
    
    # 检查是否包含所有工序
    if set(schedule) != set(range(num_operations)):
        return False
    
    return True


if __name__ == "__main__":
    # 测试调度器
    scheduler = JobShopScheduler()
    scheduler.create_problem_instance()
    
    print("作业车间调度问题实例:")
    print("=" * 40)
    
    for job in scheduler.jobs:
        print(f"工件{job.job_id}:")
        for op in job.operations:
            print(f"  工序{op.operation_id}: 机器{op.machine_id}, 时间{op.processing_time}")
    
    # 测试随机调度
    print("\n测试随机调度方案:")
    random_schedule = create_random_schedule()
    print(f"随机调度序列: {random_schedule}")
    
    if validate_schedule(random_schedule):
        scheduler.print_schedule(random_schedule)
    else:
        print("调度序列无效!")
