# 课堂小作业-遗传算法实验

本项目包含两个遗传算法实验，分别解决函数优化问题和作业车间调度问题。
课堂小作业

## 项目结构

```
遗传算法实验/
├── experiment1.py          # 实验一：函数优化主程序
├── experiment2.py          # 实验二：作业车间调度主程序
├── 1.py                   # 实验一简化版本
├── 2.py                   # 实验二简化版本
├── 第二题.py              # 实验二版本2
├── results-1/             # 实验一结果图表
│   ├── detailed_analysis.png
│   ├── evolution_process.png
│   └── objective_function.png
└── results-2/             # 实验二结果图表
    ├── detailed_analysis.png
    ├── evolution_process.png
    └── gantt_chart.png
```

## 实验一：函数优化问题

**目标**：求解函数 y = x^5 - 1000 的最大值，其中 x 为 [0, 31] 间的整数

### 算法特点
- **编码方式**：5位二进制编码
- **选择策略**：轮盘赌选择
- **交叉操作**：单点交叉
- **变异操作**：位翻转变异
- **精英策略**：保留最优个体

### 运行方法
```bash
python experiment1.py
```

### 输出结果
- 进化过程可视化
- 目标函数曲线图
- 算法收敛分析
- 最优解统计信息

## 实验二：作业车间调度问题

**目标**：3个工件在3台机器上的调度优化，最小化最大完工时间

### 问题描述
- 3个工件，每个工件3道工序
- 3台机器，每道工序有指定的加工机器和加工时间
- 约束：工序顺序约束、机器独占约束

### 算法特点
- **编码方式**：作业序列编码
- **选择策略**：锦标赛选择
- **交叉操作**：PPX交叉（保持工序顺序）
- **变异操作**：约束满足的变异
- **解码器**：基于机器可用时间的调度生成

### 运行方法
```bash
python experiment2.py
```

### 输出结果
- 进化过程可视化
- 甘特图调度方案
- 算法性能分析
- 最优调度统计

## 环境要求

- Python 3.7+
- NumPy
- Matplotlib

## 安装依赖

```bash
pip install numpy matplotlib
```

## 快速开始

1. 克隆项目
```bash
git clone https://github.com/ttiee/GeneticAlgorithm-experiment.git
cd 遗传算法实验
```

2. 安装依赖
```bash
pip install numpy matplotlib
```

3. 运行实验
```bash
# 运行实验一
python experiment1.py

# 运行实验二
python experiment2.py
```

## 结果说明

### 实验一结果
- `evolution_process.png`：算法收敛过程
- `objective_function.png`：目标函数与最优解
- `detailed_analysis.png`：详细性能分析

### 实验二结果
- `evolution_process.png`：适应度进化曲线
- `gantt_chart.png`：最优调度甘特图
- `detailed_analysis.png`：算法性能统计

## 算法参数

### 实验一参数
- 种群大小：20
- 最大迭代次数：50
- 交叉概率：0.8
- 变异概率：0.1

### 实验二参数
- 种群大小：50
- 最大迭代次数：100
- 交叉概率：0.8
- 变异概率：0.1

## 技术特点

1. **面向对象设计**：清晰的类结构，便于扩展
2. **可视化分析**：丰富的图表展示算法性能
3. **约束处理**：针对调度问题的特殊约束处理
4. **性能优化**：高效的算法实现
5. **结果保存**：自动保存图表和统计信息

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License
