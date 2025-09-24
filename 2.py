# -*- coding: utf-8 -*-
"""
遗传算法实验二主程序：作业车间调度（3工件×3机器）
------------------------------------------------
题干：

3个工件(j1, j2, j3)需要在3台机器(m1, m2, m3)上进行加工。
每个工件分别有3个工序，各工序加工所需的机器及其加工时间如下（Markdown 表）：

| 工件 | 工序1时间 | 工序2时间 | 工序3时间 | 工序1机器 | 工序2机器 | 工序3机器 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| j1   | 3         | 9         | 2         | m1        | m2        | m3        |
| j2   | 1         | 5         | 7         | m1        | m3        | m2        |
| j3   | 3         | 2         | 3         | m2        | m1        | m3        |

重要约束：
1) 每个工件的工序必须按照预定的顺序执行（O_j1 → O_j2 → O_j3）。
2) 每台机器同一时间只能加工一道工序（不可重叠）。

目标：最小化最大完工时间（makespan）。

实现要点：
- 染色体编码：长度9的作业序列（例如 [2,1,1,1,2,2,3,3,3]），每出现一次即解码为该作业的下一道工序。
- 解码器：依据作业准备时间与机器可用时间，计算每道工序的最早开工时间，更新机器完工时间与作业完工时间。
- 选择算子：锦标赛选择（tournament）。
- 交叉算子：PPX（Precedence Preservative Crossover）。
- 变异算子：随机交换两个位置（swap mutation）。
- 可视化：甘特图（matplotlib）。

"""

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 问题数据（按题干固定）
# m1=0, m2=1, m3=2
# 每个工件的三道工序：[(加工时间, 机器编号), ...]
JOBS: Dict[int, List[Tuple[int, int]]] = {
    1: [(3, 0), (9, 1), (2, 2)],  # j1: (3 on m1), (9 on m2), (2 on m3)
    2: [(1, 0), (5, 2), (7, 1)],  # j2: (1 on m1), (5 on m3), (7 on m2)
    3: [(3, 1), (2, 0), (3, 2)],  # j3: (3 on m2), (2 on m1), (3 on m3)
}
NUM_MACHINES = 3
OPS_PER_JOB = 3
GENE_TEMPLATE = [1, 1, 1, 2, 2, 2, 3, 3, 3]  # 作业序列模版


@dataclass
class OperationRecord:
    job: int
    op_index: int     # 0-based in that job
    machine: int
    start: int
    end: int


@dataclass
class Schedule:
    ops: List[OperationRecord]
    makespan: int


# 解码器：将作业序列 → 具体日程（可行）
def decode_sequence(seq: List[int]) -> Schedule:
    # job_next_op[j] = 下一道要执行的工序序号
    job_next_op = {j: 0 for j in JOBS.keys()}
    # job_ready_time[j] = 该作业可开始下一道工序的时间（上一道工序完工时间）
    job_ready_time = {j: 0 for j in JOBS.keys()}
    # machine_ready_time[m] = 机器最早可用时间
    machine_ready_time = {m: 0 for m in range(NUM_MACHINES)}

    ops_timeline: List[OperationRecord] = []

    for job in seq:
        k = job_next_op[job]  # 该作业的第k道工序
        if k >= OPS_PER_JOB:
            # 序列中该作业出现次数>3则忽略（正常不会发生）
            continue

        duration, machine = JOBS[job][k]
        est = max(job_ready_time[job], machine_ready_time[machine])  # earliest start time
        start = est
        end = start + duration

        # 记录
        ops_timeline.append(OperationRecord(job=job, op_index=k, machine=machine, start=start, end=end))

        # 更新可用时间
        job_next_op[job] += 1
        job_ready_time[job] = end
        machine_ready_time[machine] = end

    makespan = max(op.end for op in ops_timeline) if ops_timeline else 0
    return Schedule(ops=ops_timeline, makespan=makespan)


# 适应度：即 makespan（越小越好）
def fitness(seq: List[int]) -> int:
    return decode_sequence(seq).makespan

def random_individual() -> List[int]:
    s = GENE_TEMPLATE.copy()
    random.shuffle(s)
    return s

def tournament_select(pop: List[List[int]], k: int = 3) -> List[int]:
    cand = random.sample(pop, k)
    cand.sort(key=fitness)
    return cand[0][:]  # 最优者的拷贝

# PPX 交叉（Precedence Preservative Crossover）
def ppx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    needed = {1: OPS_PER_JOB, 2: OPS_PER_JOB, 3: OPS_PER_JOB}
    i, j = 0, 0
    child: List[int] = []
    # mask 控制从谁处取基因
    mask = [random.randint(0, 1) for _ in range(len(p1))]

    while len(child) < len(p1):
        if mask[len(child)] == 0:
            # 从 p1 取下一个仍需要的作业
            while i < len(p1) and needed[p1[i]] == 0:
                i += 1
            if i < len(p1):
                job = p1[i]
                child.append(job)
                needed[job] -= 1
                i += 1
                continue

        # 从 p2 取
        while j < len(p2) and needed[p2[j]] == 0:
            j += 1
        if j < len(p2):
            job = p2[j]
            child.append(job)
            needed[job] -= 1
            j += 1
            continue

        # 兜底（理论上不会到这）
        for job in [1, 2, 3]:
            while needed[job] > 0 and len(child) < len(p1):
                child.append(job)
                needed[job] -= 1

    return child

# 变异：随机交换两个基因位置
def swap_mutation(seq: List[int], pm: float) -> None:
    if random.random() < pm:
        a, b = random.sample(range(len(seq)), 2)
        seq[a], seq[b] = seq[b], seq[a]

def run_ga(
    pop_size: int = 60,
    generations: int = 200,
    pc: float = 0.9,
    pm: float = 0.2,
    tournament_k: int = 3,
    seed: int = 42,
):
    random.seed(seed)

    # 初始化群体
    population = [random_individual() for _ in range(pop_size)]
    # 记录全局最优
    best = min(population, key=fitness)
    best_fit = fitness(best)
    history = [best_fit]

    for gen in range(generations):
        new_pop: List[List[int]] = []

        # 精英保留：保留当前最优
        elite = min(population, key=fitness)
        new_pop.append(elite[:])

        # 生成下一代
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, k=tournament_k)
            p2 = tournament_select(population, k=tournament_k)
            if random.random() < pc:
                c = ppx_crossover(p1, p2)
            else:
                # 无交叉则复制父代较优者
                c = p1[:] if fitness(p1) <= fitness(p2) else p2[:]
            swap_mutation(c, pm)
            new_pop.append(c)

        population = new_pop

        # 更新全局最优
        gen_best = min(population, key=fitness)
        gen_fit = fitness(gen_best)
        if gen_fit < best_fit:
            best, best_fit = gen_best[:], gen_fit
        history.append(best_fit)

    return best, best_fit, history

def plot_gantt(schedule: Schedule, title: str = "JSSP 3×3 GA Schedule"):
    # 为每个工件指定固定颜色（一致性）
    job_colors = {
        1: "#1f77b4",  # j1
        2: "#ff7f0e",  # j2
        3: "#2ca02c",  # j3
    }

    # 按机器分组
    by_machine: Dict[int, List[OperationRecord]] = {m: [] for m in range(NUM_MACHINES)}
    for op in schedule.ops:
        by_machine[op.machine].append(op)

    # 每个机器按开始时间排序
    for m in by_machine:
        by_machine[m].sort(key=lambda x: x.start)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    yticks = []
    yticklabels = []
    bar_height = 0.8

    for m in range(NUM_MACHINES):
        ops = by_machine[m]
        y = NUM_MACHINES - 1 - m  # 机器从上到下显示
        yticks.append(y)
        yticklabels.append(f"m{m+1}")

        for op in ops:
            color = job_colors.get(op.job, None)
            ax.barh(
                y, op.end - op.start, left=op.start, height=bar_height,
                edgecolor="black", color=color
            )
            ax.text(op.start + (op.end - op.start) / 2, y,
                    f"j{op.job}-O{op.op_index+1}",
                    ha="center", va="center", fontsize=9)

    # 图例（按工件）
    handles = [plt.Rectangle((0, 0), 1, 1, color=job_colors[j], ec="black") for j in sorted(job_colors)]
    labels = [f"工件{j}" for j in sorted(job_colors)]
    ax.legend(handles, labels, title="工件", ncol=len(labels), loc="upper right", frameon=True)

    ax.set_xlabel("时间")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(f"{title}  |  最大完工时间 = {schedule.makespan}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def main():
    best_seq, best_make, hist = run_ga(
        pop_size=80,
        generations=250,
        pc=0.9,
        pm=0.2,
        tournament_k=3,
        seed=2025,
    )
    sched = decode_sequence(best_seq)

    print("=== GA 结果（实验二：3×3 JSSP）===")
    print("最佳染色体（作业序列）:", best_seq)
    print("最佳目标值（makespan）:", best_make)
    print("\n详细日程（按解码顺序）：")
    for rec in sched.ops:
        print(f"Job j{rec.job}  Op O{rec.op_index+1}  on m{rec.machine+1}  "
              f"[{rec.start}, {rec.end})  dur={rec.end - rec.start}")

    # 绘制甘特图
    plot_gantt(sched, title="基于遗传算法的3×3作业车间调度")


if __name__ == "__main__":
    main()
