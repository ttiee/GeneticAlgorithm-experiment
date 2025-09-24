import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import copy

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Operation:
    def __init__(self, job_id, operation_id, machine_id, processing_time):
        self.job_id = job_id
        self.operation_id = operation_id
        self.machine_id = machine_id
        self.processing_time = processing_time
        self.start_time = 0
        self.end_time = 0

class Job:
    def __init__(self, job_id, operations):
        self.job_id = job_id
        self.operations = operations
        self.completion_time = 0

class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.operations = []
        self.completion_time = 0

class JobShopScheduler:
    def __init__(self):
        self.jobs = []
        self.machines = []
        self.operations = []
        
    def create_problem_instance(self):
        self.jobs = []
        self.machines = []
        self.operations = []
        
        for i in range(3):
            self.machines.append(Machine(i+1))
        
        job1_ops = [Operation(1, 1, 1, 3), Operation(1, 2, 2, 9), Operation(1, 3, 3, 2)]
        job2_ops = [Operation(2, 1, 1, 1), Operation(2, 2, 3, 5), Operation(2, 3, 2, 7)]
        job3_ops = [Operation(3, 1, 2, 3), Operation(3, 2, 1, 2), Operation(3, 3, 3, 3)]
        
        self.jobs = [Job(1, job1_ops), Job(2, job2_ops), Job(3, job3_ops)]
        
        for job in self.jobs:
            self.operations.extend(job.operations)
    
    def validate_schedule_constraints(self, schedule):
        job_operation_order = {1: [], 2: [], 3: []}
        
        for operation_id in schedule:
            operation = self.operations[operation_id]
            job_id = operation.job_id
            operation_seq = operation.operation_id
            job_operation_order[job_id].append(operation_seq)
        
        for job_id in [1, 2, 3]:
            operations = job_operation_order[job_id]
            if operations != sorted(operations):
                return False
        return True
    
    def calculate_makespan(self, schedule):
        if not self.validate_schedule_constraints(schedule):
            return 999999
        
        for job in self.jobs:
            job.completion_time = 0
            for op in job.operations:
                op.start_time = 0
                op.end_time = 0
        
        for machine in self.machines:
            machine.completion_time = 0
            machine.operations = []
        
        for operation_id in schedule:
            operation = self.operations[operation_id]
            job = self.jobs[operation.job_id - 1]
            machine = self.machines[operation.machine_id - 1]
            
            prev_operation_end = 0
            if operation.operation_id > 1:
                prev_operation = job.operations[operation.operation_id - 2]
                prev_operation_end = prev_operation.end_time
            
            machine_available_time = machine.completion_time
            operation.start_time = max(prev_operation_end, machine_available_time)
            operation.end_time = operation.start_time + operation.processing_time
            
            machine.completion_time = operation.end_time
            machine.operations.append(operation)
            job.completion_time = max(job.completion_time, operation.end_time)
        
        return max(job.completion_time for job in self.jobs)
    
    def get_schedule_info(self, schedule):
        makespan = self.calculate_makespan(schedule)
        job_completion_times = {job.job_id: job.completion_time for job in self.jobs}
        machine_completion_times = {machine.machine_id: machine.completion_time for machine in self.machines}
        
        return {
            'makespan': makespan,
            'job_completion_times': job_completion_times,
            'machine_completion_times': machine_completion_times,
            'schedule': schedule
        }

class JobShopIndividual:
    def __init__(self, chromosome=None, num_operations=9):
        if chromosome is None:
            self.chromosome = list(range(num_operations))
            random.shuffle(self.chromosome)
        else:
            self.chromosome = chromosome.copy()
        
        self.fitness = 0.0
        self.makespan = 0
    
    def calculate_fitness(self, scheduler):
        self.makespan = scheduler.calculate_makespan(self.chromosome)
        self.fitness = 1.0 / (self.makespan + 1)
        return self.fitness

class JobShopPopulation:
    def __init__(self, size, num_operations=9):
        self.size = size
        self.num_operations = num_operations
        self.individuals = self._create_constraint_satisfying_population(size, num_operations)
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.best_fitness_history = []
        self.average_fitness_history = []
    
    def _create_constraint_satisfying_population(self, size, num_operations):
        individuals = []
        
        for _ in range(size):
            job1_ops = [0, 1, 2]
            job2_ops = [3, 4, 5]
            job3_ops = [6, 7, 8]
            
            random.shuffle(job1_ops)
            random.shuffle(job2_ops)
            random.shuffle(job3_ops)
            
            all_operations = job1_ops + job2_ops + job3_ops
            random.shuffle(all_operations)
            
            schedule = self._create_valid_schedule(all_operations)
            individuals.append(JobShopIndividual(schedule))
        
        return individuals
    
    def _create_valid_schedule(self, operations):
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        job_operations = {1: [], 2: [], 3: []}
        for op in operations:
            job_id = operation_to_job[op]
            job_operations[job_id].append(op)
        
        for job_id in [1, 2, 3]:
            job_operations[job_id].sort(key=lambda x: operation_to_seq[x])
        
        valid_schedule = []
        remaining_ops = operations.copy()
        
        while remaining_ops:
            available_ops = []
            
            for op in remaining_ops:
                job_id = operation_to_job[op]
                seq = operation_to_seq[op]
                
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
            
            if available_ops:
                selected_op = random.choice(available_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
            else:
                selected_op = random.choice(remaining_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
        
        return valid_schedule
    
    def evaluate_population(self, scheduler):
        for individual in self.individuals:
            individual.calculate_fitness(scheduler)
        
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or self.individuals[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.individuals[0])
        
        fitnesses = [ind.fitness for ind in self.individuals]
        self.fitness_history.append(fitnesses)
        self.best_fitness_history.append(self.best_individual.fitness)
        self.average_fitness_history.append(np.mean(fitnesses))
    
    def get_best_individual(self):
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_average_fitness(self):
        return np.mean([ind.fitness for ind in self.individuals])

class JobShopGeneticAlgorithm:
    def __init__(self, population_size=50, crossover_rate=0.8, mutation_rate=0.1, max_generations=100, num_operations=9):
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
    
    def selection(self, population):
        total_fitness = sum(ind.fitness for ind in population.individuals)
        if total_fitness == 0:
            return random.choices(population.individuals, k=population.size)
        
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
    
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1_chromosome = self._create_constraint_satisfying_offspring(parent1.chromosome, parent2.chromosome)
        child2_chromosome = self._create_constraint_satisfying_offspring(parent2.chromosome, parent1.chromosome)
        
        return JobShopIndividual(child1_chromosome), JobShopIndividual(child2_chromosome)
    
    def _create_constraint_satisfying_offspring(self, parent1, parent2):
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        job_inheritance = {}
        for job_id in [1, 2, 3]:
            job_inheritance[job_id] = random.choice([parent1, parent2])
        
        job_operations = {1: [], 2: [], 3: []}
        for op in range(9):
            job_id = operation_to_job[op]
            job_operations[job_id].append(op)
        
        offspring_operations = []
        
        for job_id in [1, 2, 3]:
            parent = job_inheritance[job_id]
            
            job_ops_in_parent = []
            for op in parent:
                if operation_to_job[op] == job_id:
                    job_ops_in_parent.append(op)
            
            job_ops_in_parent.sort(key=lambda x: operation_to_seq[x])
            offspring_operations.extend(job_ops_in_parent)
        
        return self._create_valid_schedule_from_operations(offspring_operations)
    
    def _create_valid_schedule_from_operations(self, operations):
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        valid_schedule = []
        remaining_ops = operations.copy()
        
        while remaining_ops:
            available_ops = []
            
            for op in remaining_ops:
                job_id = operation_to_job[op]
                seq = operation_to_seq[op]
                
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
            
            if available_ops:
                selected_op = random.choice(available_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
            else:
                selected_op = random.choice(remaining_ops)
                valid_schedule.append(selected_op)
                remaining_ops.remove(selected_op)
        
        return valid_schedule
    
    def mutation(self, individual):
        if random.random() > self.mutation_rate:
            return copy.deepcopy(individual)
        
        mutated_chromosome = self._create_constraint_satisfying_mutation(individual.chromosome)
        return JobShopIndividual(mutated_chromosome)
    
    def _create_constraint_satisfying_mutation(self, chromosome):
        operation_to_job = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
        operation_to_seq = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3}
        
        job_operations = {1: [], 2: [], 3: []}
        for op in chromosome:
            job_id = operation_to_job[op]
            job_operations[job_id].append(op)
        
        for job_id in [1, 2, 3]:
            job_operations[job_id].sort(key=lambda x: operation_to_seq[x])
        
        mutation_type = random.choice(['swap_jobs', 'reorder_within_job', 'random_reschedule'])
        
        if mutation_type == 'swap_jobs':
            job1, job2 = random.sample([1, 2, 3], 2)
            job_operations[job1], job_operations[job2] = job_operations[job2], job_operations[job1]
        elif mutation_type == 'random_reschedule':
            all_operations = job_operations[1] + job_operations[2] + job_operations[3]
            return self._create_valid_schedule_from_operations(all_operations)
        
        all_operations = job_operations[1] + job_operations[2] + job_operations[3]
        return self._create_valid_schedule_from_operations(all_operations)
    
    def evolve_generation(self):
        selected_individuals = self.selection(self.population)
        
        new_individuals = []
        
        if self.population.best_individual:
            new_individuals.append(copy.deepcopy(self.population.best_individual))
        
        for i in range(0, len(selected_individuals) - 1, 2):
            parent1, parent2 = selected_individuals[i], selected_individuals[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            new_individuals.extend([child1, child2])
        
        while len(new_individuals) < self.population_size:
            new_individuals.append(JobShopIndividual(num_operations=self.num_operations))
        
        self.population.individuals = new_individuals[:self.population_size]
        self.population.generation += 1
        self.generation += 1
    
    def run(self):
        print("开始作业车间调度遗传算法优化...")
        
        self.population.evaluate_population(self.scheduler)
        
        for generation in range(self.max_generations):
            self.evolve_generation()
            self.population.evaluate_population(self.scheduler)
            
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
            
            if (generation + 1) % 10 == 0 or generation == 0:
                best_ind = self.population.get_best_individual()
                print(f"第 {generation + 1:3d} 代: 最佳适应度 = {best_fitness:8.4f}, "
                      f"平均适应度 = {avg_fitness:8.4f}, 最小完工时间 = {best_makespan:2d}")
        
        print("优化完成!")
        best_solution = self.population.get_best_individual()
        print(f"最优解: 完工时间 = {best_solution.makespan}, 适应度 = {best_solution.fitness:.4f}")
        
        return (best_solution, 
                self.population.best_fitness_history,
                self.population.average_fitness_history,
                self.convergence_history)
    
    def get_scheduler(self):
        return self.scheduler

class JobShopVisualizer:
    def __init__(self):
        self.fig_size = (15, 10)
        self.dpi = 100
    
    def plot_evolution_process(self, best_fitness_history, average_fitness_history, save_path=None):
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
    
    def plot_detailed_analysis(self, convergence_history, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size, dpi=self.dpi)
        fig.suptitle('作业车间调度遗传算法详细分析', fontsize=16, fontweight='bold')
        
        generations = [conv['generation'] for conv in convergence_history]
        best_fitnesses = [conv['best_fitness'] for conv in convergence_history]
        avg_fitnesses = [conv['average_fitness'] for conv in convergence_history]
        best_makespans = [conv['best_makespan'] for conv in convergence_history]
        
        ax1 = axes[0, 0]
        ax1.plot(generations, best_fitnesses, 'r-', linewidth=2, label='最佳适应度')
        ax1.plot(generations, avg_fitnesses, 'b-', linewidth=2, label='平均适应度')
        ax1.fill_between(generations, avg_fitnesses, best_fitnesses, alpha=0.3, color='green')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('适应度值')
        ax1.set_title('适应度对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(generations, best_makespans, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('完工时间')
        ax2.set_title('最优完工时间变化')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        if len(best_fitnesses) > 1:
            convergence_rate = np.diff(best_fitnesses)
            ax3.plot(generations[1:], convergence_rate, 'purple', linewidth=2, marker='^')
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('适应度变化')
            ax3.set_title('收敛性分析')
            ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
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
    
    def plot_gantt_chart(self, schedule, scheduler, save_path=None):
        info = scheduler.get_schedule_info(schedule)
        
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
        
        operations.sort(key=lambda x: x['start_time'])
        
        fig, ax = plt.subplots(figsize=(15, 8), dpi=self.dpi)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        machine_y_positions = {1: 0, 2: 1, 3: 2}
        machine_names = {1: '机器1', 2: '机器2', 3: '机器3'}
        
        for i, op in enumerate(operations):
            machine_id = op['machine_id']
            y_pos = machine_y_positions[machine_id]
            
            color = colors[op['job_id'] - 1]
            ax.barh(y_pos, op['processing_time'], 
                   left=op['start_time'], height=0.6, 
                   color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            ax.text(op['start_time'] + op['processing_time']/2, y_pos, 
                   f'J{op["job_id"]}O{op["operation_id"]}', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['机器1', '机器2', '机器3'])
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('机器', fontsize=12)
        ax.set_title(f'作业车间调度甘特图 (完工时间: {info["makespan"]})', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        
        legend_elements = []
        for i in range(3):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                               facecolor=colors[i], 
                                               label=f'工件{i+1}'))
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlim(0, info['makespan'] + 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"甘特图已保存到: {save_path}")
        
        plt.show()

def run_experiment():
    print("=" * 60)
    print("遗传算法实验二: 作业车间调度问题")
    print("=" * 60)
    
    config = {
        'population_size': 50,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'max_generations': 100
    }
    
    print("算法参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    ga = JobShopGeneticAlgorithm(
        population_size=config['population_size'],
        crossover_rate=config['crossover_rate'],
        mutation_rate=config['mutation_rate'],
        max_generations=config['max_generations']
    )
    
    best_solution, best_fitness_history, avg_fitness_history, convergence_history = ga.run()
    
    print(f"最优解: 完工时间 = {best_solution.makespan}")
    
    scheduler = ga.get_scheduler()
    visualizer = JobShopVisualizer()
    
    output_dir = "results-2"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n正在生成可视化图表...")
    
    visualizer.plot_evolution_process(
        best_fitness_history, 
        avg_fitness_history,
        save_path=os.path.join(output_dir, "evolution_process.png")
    )
    
    visualizer.plot_detailed_analysis(
        convergence_history,
        save_path=os.path.join(output_dir, "detailed_analysis.png")
    )
    
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
        'scheduler': scheduler
    }

if __name__ == "__main__":
    results = run_experiment()
    print("\n实验完成!")