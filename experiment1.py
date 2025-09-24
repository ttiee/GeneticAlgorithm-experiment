"""
遗传算法实验一主程序

题干：
求解函数 y = x^5 - 1000 的最大值，其中 x 为 [0, 31] 间的整数
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import copy
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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


def objective_function(x: int) -> float:
    """
    目标函数: y = x^5 - 1000
    
    Args:
        x: 输入值 (0-31之间的整数)
    
    Returns:
        函数值
    """
    return x ** 5 - 1000


def plot_objective_function(best_solution, save_path=None):
    """
    绘制目标函数曲线和最优解
    
    Args:
        best_solution: 最优解个体
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8), dpi=100)
    
    # 计算目标函数值
    x_range = np.arange(0, 32)
    y_values = [objective_function(x) for x in x_range]
    
    # 绘制目标函数曲线
    plt.plot(x_range, y_values, 'b-', linewidth=3, label='目标函数 y = x^5 - 1000', marker='o', markersize=6)
    
    # 标记最优解
    plt.plot(best_solution.value, best_solution.fitness, 'ro', markersize=15, 
            label=f'遗传算法最优解: x={best_solution.value}, y={best_solution.fitness:.2f}')
    
    # 标记理论最优解
    theoretical_max = max(y_values)
    theoretical_x = x_range[np.argmax(y_values)]
    plt.plot(theoretical_x, theoretical_max, 'g^', markersize=15, 
            label=f'理论最优解: x={theoretical_x}, y={theoretical_max:.2f}')
    
    plt.xlabel('x 值', fontsize=12)
    plt.ylabel('y 值', fontsize=12)
    plt.title('目标函数 y = x^5 - 1000 与最优解', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    plt.xlim(-1, 32)
    plt.ylim(min(y_values) - 1000, max(y_values) + 1000)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"目标函数图已保存到: {save_path}")
    
    plt.show()


def plot_simple_evolution(best_fitness_history, average_fitness_history, save_path=None):
    """
    绘制简单的进化过程图
    
    Args:
        best_fitness_history: 最佳适应度历史
        average_fitness_history: 平均适应度历史
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8), dpi=100)
    
    generations = list(range(1, len(best_fitness_history) + 1))
    
    plt.plot(generations, best_fitness_history, 'r-', linewidth=3, 
            label='最佳适应度', marker='o', markersize=6)
    plt.plot(generations, average_fitness_history, 'b-', linewidth=3, 
            label='平均适应度', marker='s', markersize=6)
    
    plt.fill_between(generations, average_fitness_history, best_fitness_history, 
                    alpha=0.2, color='green', label='适应度范围')
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度值', fontsize=12)
    plt.title('遗传算法进化过程', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加最优解信息
    if best_fitness_history:
        final_best = max(best_fitness_history)
        plt.annotate(f'最优适应度: {final_best:.2f}', 
                    xy=(len(generations), final_best), 
                    xytext=(len(generations) * 0.7, final_best * 1.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"进化过程图已保存到: {save_path}")
    
    plt.show()


def plot_evolution_process(best_fitness_history, average_fitness_history, convergence_history, save_path=None):
    """
    绘制进化过程图
    
    Args:
        best_fitness_history: 最佳适应度历史
        average_fitness_history: 平均适应度历史
        convergence_history: 收敛历史详细信息
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    fig.suptitle('遗传算法进化过程分析', fontsize=16, fontweight='bold')
    
    generations = list(range(1, len(best_fitness_history) + 1))
    
    # 1. 适应度进化曲线
    ax1 = axes[0, 0]
    ax1.plot(generations, best_fitness_history, 'r-', linewidth=2, label='最佳适应度', marker='o', markersize=4)
    ax1.plot(generations, average_fitness_history, 'b-', linewidth=2, label='平均适应度', marker='s', markersize=4)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('适应度值')
    ax1.set_title('适应度进化曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 适应度分布箱线图
    ax2 = axes[0, 1]
    fitness_data = []
    for conv in convergence_history:
        if 'fitness_distribution' in conv:
            fitness_data.append(conv['fitness_distribution'])
    
    if fitness_data:
        ax2.boxplot(fitness_data, positions=generations[:len(fitness_data)])
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('适应度分布')
        ax2.set_title('适应度分布箱线图')
        ax2.grid(True, alpha=0.3)
    else:
        # 如果没有分布数据，显示适应度变化
        ax2.plot(generations, best_fitness_history, 'g-', linewidth=2, marker='^', markersize=4)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('最佳适应度')
        ax2.set_title('最佳适应度变化')
        ax2.grid(True, alpha=0.3)
    
    # 3. 收敛速度分析
    ax3 = axes[1, 0]
    if len(best_fitness_history) > 1:
        improvement = np.diff(best_fitness_history)
        ax3.plot(generations[1:], improvement, 'g-', linewidth=2, marker='^', markersize=4)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('适应度改进量')
        ax3.set_title('收敛速度分析')
        ax3.grid(True, alpha=0.3)
    
    # 4. 目标函数曲线和最优解
    ax4 = axes[1, 1]
    x_range = np.arange(0, 32)
    y_values = [x**5 - 1000 for x in x_range]
    
    ax4.plot(x_range, y_values, 'b-', linewidth=2, label='目标函数 y = x^5 - 1000')
    
    # 标记最优解
    if convergence_history:
        best_solution = convergence_history[-1]['best_individual']
        ax4.plot(best_solution.value, best_solution.fitness, 'ro', markersize=10, 
                label=f'最优解: x={best_solution.value}, y={best_solution.fitness:.2f}')
    
    ax4.set_xlabel('x 值')
    ax4.set_ylabel('y 值')
    ax4.set_title('目标函数与最优解')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"进化过程图已保存到: {save_path}")
    
    plt.show()


def run_experiment():
    """运行遗传算法实验"""
    print("=" * 60)
    print("遗传算法实验一: 求解函数 y = x^5 - 1000 的最大值")
    print("=" * 60)
    
    # 算法参数配置
    config = {
        'population_size': 5,      # 种群大小
        'chromosome_length': 5,     # 染色体长度 (5位二进制表示0-31)
        'crossover_rate': 0.8,      # 交叉概率
        'mutation_rate': 0.1,       # 变异概率
        'max_generations': 50       # 最大迭代次数
    }
    
    print("算法参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建遗传算法实例
    ga = GeneticAlgorithm(
        population_size=config['population_size'],
        chromosome_length=config['chromosome_length'],
        crossover_rate=config['crossover_rate'],
        mutation_rate=config['mutation_rate'],
        max_generations=config['max_generations'],
        objective_function=objective_function
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行遗传算法
    best_solution, best_fitness_history, avg_fitness_history, convergence_history = ga.run()
    
    # 记录结束时间
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n算法执行时间: {execution_time:.2f} 秒")
    print(f"最优解: x = {best_solution.value}, y = {best_solution.fitness:.2f}")
    
    # 理论最优解验证
    theoretical_max = max([objective_function(x) for x in range(32)])
    theoretical_x = max(range(32), key=lambda x: objective_function(x))
    print(f"理论最优解: x = {theoretical_x}, y = {theoretical_max:.2f}")
    
    # 创建输出目录
    output_dir = "results-1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 绘制进化过程图
    print("\n正在生成可视化图表...")
    
    # 1. 简单进化过程图
    plot_simple_evolution(
        best_fitness_history, 
        avg_fitness_history,
        save_path=os.path.join(output_dir, "evolution_process.png")
    )
    
    # 2. 详细分析图
    plot_evolution_process(
        best_fitness_history,
        avg_fitness_history, 
        convergence_history,
        save_path=os.path.join(output_dir, "detailed_analysis.png")
    )
    
    # 3. 目标函数可视化
    plot_objective_function(best_solution, save_path=os.path.join(output_dir, "objective_function.png"))
    
    print(f"\n所有图表已保存到 '{output_dir}' 目录")
    
    return {
        'best_solution': best_solution,
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'convergence_history': convergence_history,
        'execution_time': execution_time,
        'theoretical_optimal': (theoretical_x, theoretical_max)
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
        'population_size': 30,
        'chromosome_length': 5,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'max_generations': 50
    }
    
    for run in range(num_runs):
        print(f"\n第 {run + 1} 次实验:")
        print("-" * 30)
        
        ga = GeneticAlgorithm(
            population_size=config['population_size'],
            chromosome_length=config['chromosome_length'],
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate'],
            max_generations=config['max_generations'],
            objective_function=objective_function
        )
        
        best_solution, _, _, _ = ga.run()
        results.append({
            'run': run + 1,
            'best_x': best_solution.value,
            'best_fitness': best_solution.fitness
        })
    
    # 分析结果
    print("\n" + "=" * 50)
    print("多次实验结果:")
    print("=" * 50)
    
    best_fitnesses = [r['best_fitness'] for r in results]
    best_x_values = [r['best_x'] for r in results]
    
    print(f"最佳适应度统计:")
    print(f"  平均值: {np.mean(best_fitnesses):.2f}")
    print(f"  标准差: {np.std(best_fitnesses):.2f}")
    print(f"  最大值: {np.max(best_fitnesses):.2f}")
    print(f"  最小值: {np.min(best_fitnesses):.2f}")
    
    print(f"\n最优解 x 值统计:")
    print(f"  平均值: {np.mean(best_x_values):.2f}")
    print(f"  标准差: {np.std(best_x_values):.2f}")
    print(f"  最频繁值: {max(set(best_x_values), key=best_x_values.count)}")
    
    # 成功率分析
    theoretical_max = max([objective_function(x) for x in range(32)])
    success_count = sum(1 for f in best_fitnesses if abs(f - theoretical_max) < 1e-6)
    success_rate = success_count / num_runs * 100
    
    print(f"\n算法性能:")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"  平均误差: {np.mean([abs(f - theoretical_max) for f in best_fitnesses]):.2f}")


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
