"""
遗传算法实验二主程序
求解作业车间调度问题：3个工件在3台机器上的调度优化
"""

import numpy as np
import matplotlib.pyplot as plt
from job_shop_ga import JobShopGeneticAlgorithm
from job_shop_scheduling import JobShopScheduler
import time
import os

#设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
    
    # 算法参数配置
    config = {
        'population_size': 50,      # 种群大小
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
