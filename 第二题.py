import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

jobs_data = {
    'j1': [('m1', 3), ('m2', 9), ('m3', 2)],
    'j2': [('m1', 1), ('m3', 5), ('m2', 7)],
    'j3': [('m2', 3), ('m1', 2), ('m3', 3)]
}

job_colors = {
    'j1': '#FF6B6B',
    'j2': '#4ECDC4',
    'j3': '#45B7D1'
}

n_jobs = 3
n_machines = 3
chromosome_length = n_jobs * n_machines

POPULATION_SIZE = 100
GENERATIONS = 20
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
ELITISM_COUNT = 3

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_chromosome():
    chromosome = []
    for _ in range(chromosome_length):
        machine_num = random.randint(1, n_machines)
        decimal_part = random.uniform(0.01, 0.99)
        gene = machine_num + decimal_part
        chromosome.append(gene)
    return chromosome


def decode_chromosome(chromosome):
    machine_sequences = {1: [], 2: [], 3: []}

    for i, gene in enumerate(chromosome):
        machine_num = int(gene)
        decimal_part = gene - machine_num
        job_num = (i % n_jobs) + 1
        machine_sequences[machine_num].append((decimal_part, f'j{job_num}', i))

    for machine in machine_sequences:
        machine_sequences[machine].sort(key=lambda x: x[0])
        machine_sequences[machine] = [(job, idx) for _, job, idx in machine_sequences[machine]]

    return machine_sequences


def calculate_worktime(chromosome):
    machine_sequences = decode_chromosome(chromosome)

    machine_available = {'m1': 0, 'm2': 0, 'm3': 0}
    job_next_operation = {'j1': 0, 'j2': 0, 'j3': 0}
    job_ready_time = {'j1': 0, 'j2': 0, 'j3': 0}

    all_operations = []
    for machine_num, sequence in machine_sequences.items():
        machine_id = f'm{machine_num}'
        for job_id, gene_index in sequence:
            operation_index = -1
            for i, (m, _) in enumerate(jobs_data[job_id]):
                if m == machine_id:
                    operation_index = i
                    break
            if operation_index != -1:
                all_operations.append((job_id, machine_id, operation_index, gene_index))

    all_operations.sort(key=lambda x: x[3])

    completed_operations = 0
    total_operations = n_jobs * 3

    operation_schedule = []

    while completed_operations < total_operations:
        progress_made = False

        for job_id, machine_id, op_index, gene_index in all_operations:
            if op_index != job_next_operation[job_id]:
                continue

            if op_index > 0:
                prev_op_completed = True
                for prev_job, prev_machine, prev_op, _ in all_operations:
                    if prev_job == job_id and prev_op == op_index - 1:
                        if prev_op >= job_next_operation[job_id]:
                            prev_op_completed = False
                            break
                if not prev_op_completed:
                    continue

            duration = jobs_data[job_id][op_index][1]
            start_time = max(machine_available[machine_id], job_ready_time[job_id])
            end_time = start_time + duration

            operation_schedule.append({
                'job': job_id,
                'machine': machine_id,
                'operation': op_index + 1,
                'start': start_time,
                'end': end_time,
                'duration': duration
            })

            machine_available[machine_id] = end_time
            job_ready_time[job_id] = end_time
            job_next_operation[job_id] += 1
            completed_operations += 1
            progress_made = True

            all_operations = [op for op in all_operations if not (op[0] == job_id and op[2] == op_index)]
            break

        if not progress_made:
            return float('inf'), []

    return max(job_ready_time.values()), operation_schedule


def calculate_fitness(chromosome):
    worktime, _ = calculate_worktime(chromosome)
    return 1.0 / (worktime + 1e-6)


def initialize_population():
    return [create_chromosome() for _ in range(POPULATION_SIZE)]


def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return random.sample(population, 2)

    probabilities = [fitness / total_fitness for fitness in fitness_values]
    cumulative_probabilities = np.cumsum(probabilities)

    selected = []
    for _ in range(2):
        rand = random.random()
        for i, cum_prob in enumerate(cumulative_probabilities):
            if rand <= cum_prob:
                selected.append(population[i])
                break
    return selected


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    length = len(parent1)
    point1 = random.randint(1, length - 2)
    point2 = random.randint(point1 + 1, length - 1)

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2


def mutate(chromosome):
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5:
                new_machine = random.randint(1, n_machines)
                decimal_part = mutated[i] - int(mutated[i])
                mutated[i] = new_machine + decimal_part
            else:
                machine_num = int(mutated[i])
                new_decimal = random.uniform(0.01, 0.99)
                mutated[i] = machine_num + new_decimal
    return mutated


def plot_convergence(best_makespan_history, avg_makespan_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    generations = range(len(best_makespan_history))
    plt.plot(generations, best_makespan_history, 'b-', linewidth=2, label='最佳完成时间')
    plt.plot(generations, avg_makespan_history, 'r--', linewidth=2, label='平均完成时间')
    plt.xlabel('迭代代数')
    plt.ylabel('完成时间')
    plt.title('遗传算法收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)

    plt.tight_layout()
    plt.show()


def plot_gantt_chart(operation_schedule, makespan):
    fig, ax = plt.subplots(figsize=(12, 6))

    machine_positions = {'m1': 0, 'm2': 1, 'm3': 2}

    for operation in operation_schedule:
        machine = operation['machine']
        job = operation['job']
        start = operation['start']
        duration = operation['duration']

        y_pos = machine_positions[machine]
        color = job_colors[job]

        rect = patches.Rectangle((start, y_pos - 0.4), duration, 0.8,
                                 linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)

        ax.text(start + duration / 2, y_pos, f'{job}\nOP{operation["operation"]}',
                ha='center', va='center', fontweight='bold', fontsize=10)

    ax.set_xlim(0, makespan + 2)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['机器1', '机器2', '机器3'])
    ax.set_xlabel('时间')
    ax.set_ylabel('机器')
    ax.set_title(f'作业车间调度甘特图 (总完成时间: {makespan})')
    ax.grid(True, alpha=0.3)

    # 添加图例
    legend_elements = [patches.Patch(color=job_colors[job], label=job)
                       for job in job_colors]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_chromosome_visualization(chromosome, makespan):
    machine_sequences = decode_chromosome(chromosome)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = 0
    for machine in sorted(machine_sequences.keys()):
        sequence = machine_sequences[machine]
        ax.text(-1, y_pos, f'机器{machine}:', ha='right', va='center', fontweight='bold')

        x_pos = 0
        for job, gene_idx in sequence:
            color = job_colors[job]
            ax.add_patch(patches.Rectangle((x_pos, y_pos - 0.3), 1, 0.6,
                                           facecolor=color, alpha=0.7, edgecolor='black'))
            ax.text(x_pos + 0.5, y_pos, job, ha='center', va='center', fontweight='bold')
            x_pos += 1.2

        y_pos += 1

    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['机器1序列', '机器2序列', '机器3序列'])
    ax.set_title(f'染色体解码可视化 (最终用时: {makespan})')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')

    plt.tight_layout()
    plt.show()


def genetic_algorithm():
    population = initialize_population()
    best_makespan_history = []
    avg_makespan_history = []

    best_chromosome = None
    best_makespan = float('inf')
    best_schedule = []

    print("开始遗传算法优化...")

    for generation in range(GENERATIONS):
        makespan_values = []
        fitness_values = []

        for chrom in population:
            makespan, _ = calculate_worktime(chrom)
            makespan_values.append(makespan)
            fitness_values.append(calculate_fitness(chrom))

        current_best = min(makespan_values)
        current_best_idx = makespan_values.index(current_best)
        current_avg = np.mean(makespan_values)

        if current_best < best_makespan:
            best_makespan = current_best
            best_chromosome = population[current_best_idx]
            _, best_schedule = calculate_worktime(best_chromosome)

        best_makespan_history.append(current_best)
        avg_makespan_history.append(current_avg)

        if generation % 20 == 0:
            print(f"代 {generation:3d}: 最佳={current_best:.1f}, 平均={current_avg:.1f}")

        new_population = []
        elite_indices = np.argsort(makespan_values)[:ELITISM_COUNT]
        for idx in elite_indices:
            new_population.append(population[idx])

        while len(new_population) < POPULATION_SIZE:
            parents = roulette_wheel_selection(population, fitness_values)
            child1, child2 = crossover(parents[0], parents[1])
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:POPULATION_SIZE]

    print(f"优化完成！最佳完成时间: {best_makespan}")

    plot_convergence(best_makespan_history, avg_makespan_history)
    plot_gantt_chart(best_schedule, best_makespan)
    plot_chromosome_visualization(best_chromosome, best_makespan)

    print_schedule(best_chromosome)

    return best_chromosome, best_makespan, best_makespan_history, best_schedule


def print_schedule(chromosome):
    worktime, schedule = calculate_worktime(chromosome)
    print(f"\n最佳调度方案 (完成时间: {worktime}):")

    # 按机器分组显示
    machine_schedules = {'m1': [], 'm2': [], 'm3': []}
    for op in schedule:
        machine_schedules[op['machine']].append(op)

    for machine in ['m1', 'm2', 'm3']:
        machine_schedules[machine].sort(key=lambda x: x['start'])
        print(f"\n{machine}机器调度:")
        print("开始-结束\t工件\t工序\t时间")
        print("-" * 40)
        for op in machine_schedules[machine]:
            print(f"{op['start']:2.0f}-{op['end']:2.0f}\t\t{op['job']}\t{op['operation']}\t{op['duration']}")


# 运行算法
if __name__ == "__main__":
    best_chromosome, best_makespan, history, schedule = genetic_algorithm()
