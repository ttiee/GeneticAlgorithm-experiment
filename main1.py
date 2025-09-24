"""
遗传算法实验一主程序

题干：
求解函数 y = x^5 - 1000 的最大值，其中 x 为 [0, 31] 间的整数
"""

import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from visualization import GAVisualizer
import time
import os


def objective_function(x: int) -> float:
    """
    目标函数: y = x^5 - 1000
    
    Args:
        x: 输入值 (0-31之间的整数)
    
    Returns:
        函数值
    """
    return x ** 5 - 1000


def run_experiment():
    """运行遗传算法实验"""
    print("=" * 60)
    print("遗传算法实验一: 求解函数 y = x^5 - 1000 的最大值")
    print("=" * 60)
    
    # 算法参数配置
    config = {
        'population_size': 30,      # 种群大小
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
    
    # 创建可视化器
    visualizer = GAVisualizer()
    
    # 创建输出目录
    output_dir = "results-1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 绘制进化过程图
    print("\n正在生成可视化图表...")
    
    # 1. 简单进化过程图
    visualizer.plot_simple_evolution(
        best_fitness_history, 
        avg_fitness_history,
        save_path=os.path.join(output_dir, "evolution_process.png")
    )
    
    # 2. 详细分析图
    visualizer.plot_evolution_process(
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
