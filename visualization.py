"""
可视化模块
用于绘制遗传算法的进化过程图
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GAVisualizer:
    """遗传算法可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.fig_size = (15, 10)
        self.dpi = 100
    
    def plot_evolution_process(self, 
                             best_fitness_history: List[float],
                             average_fitness_history: List[float],
                             convergence_history: List[Dict[str, Any]],
                             save_path: str = None):
        """
        绘制进化过程图
        
        Args:
            best_fitness_history: 最佳适应度历史
            average_fitness_history: 平均适应度历史
            convergence_history: 收敛历史详细信息
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size, dpi=self.dpi)
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
    
    def plot_detailed_analysis(self, 
                             convergence_history: List[Dict[str, Any]],
                             save_path: str = None):
        """
        绘制详细分析图
        
        Args:
            convergence_history: 收敛历史详细信息
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle('遗传算法详细分析', fontsize=16, fontweight='bold')
        
        generations = [conv['generation'] for conv in convergence_history]
        best_fitnesses = [conv['best_fitness'] for conv in convergence_history]
        avg_fitnesses = [conv['average_fitness'] for conv in convergence_history]
        
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
        
        # 2. 收敛性分析
        ax2 = axes[0, 1]
        if len(best_fitnesses) > 1:
            convergence_rate = np.diff(best_fitnesses)
            ax2.plot(generations[1:], convergence_rate, 'g-', linewidth=2, marker='o')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('适应度变化')
            ax2.set_title('收敛性分析')
            ax2.grid(True, alpha=0.3)
        
        # 3. 目标函数3D视图（简化版）
        ax3 = axes[0, 2]
        x_range = np.arange(0, 32)
        y_values = [x**5 - 1000 for x in x_range]
        ax3.bar(x_range, y_values, alpha=0.7, color='skyblue')
        if convergence_history:
            best_solution = convergence_history[-1]['best_individual']
            ax3.bar(best_solution.value, best_solution.fitness, color='red', alpha=0.8)
        ax3.set_xlabel('x 值')
        ax3.set_ylabel('y 值')
        ax3.set_title('目标函数分布')
        ax3.grid(True, alpha=0.3)
        
        # 4. 种群多样性分析（如果有数据）
        ax4 = axes[1, 0]
        if 'diversity' in convergence_history[0]:
            diversity = [conv['diversity'] for conv in convergence_history]
            ax4.plot(generations, diversity, 'purple', linewidth=2, marker='s')
            ax4.set_xlabel('迭代次数')
            ax4.set_ylabel('种群多样性')
            ax4.set_title('种群多样性变化')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '多样性数据不可用', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('种群多样性分析')
        
        # 5. 最优解进化轨迹
        ax5 = axes[1, 1]
        best_values = [conv['best_individual'].value for conv in convergence_history]
        ax5.plot(generations, best_values, 'orange', linewidth=2, marker='D')
        ax5.set_xlabel('迭代次数')
        ax5.set_ylabel('最优解 x 值')
        ax5.set_title('最优解进化轨迹')
        ax5.grid(True, alpha=0.3)
        
        # 6. 统计信息
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 计算统计信息
        final_best = convergence_history[-1]['best_fitness']
        initial_best = convergence_history[0]['best_fitness']
        improvement = final_best - initial_best
        improvement_rate = (improvement / abs(initial_best)) * 100 if initial_best != 0 else 0
        
        stats_text = f"""
        算法统计信息:
        
        初始最佳适应度: {initial_best:.2f}
        最终最佳适应度: {final_best:.2f}
        适应度改进: {improvement:.2f}
        改进率: {improvement_rate:.2f}%
        
        总迭代次数: {len(convergence_history)}
        最优解 x 值: {convergence_history[-1]['best_individual'].value}
        最优解 y 值: {final_best:.2f}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_simple_evolution(self, 
                            best_fitness_history: List[float],
                            average_fitness_history: List[float],
                            save_path: str = None):
        """
        绘制简单的进化过程图
        
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
